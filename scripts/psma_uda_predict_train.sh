#!/usr/bin/env bash
# PSMA UDA：对 splits train 病例做滑窗推理（硬分割 + probabilities）
# 多卡并行分片：默认 GPU 0,1,3，每卡 TASK1_UDA_PRED_PER_GPU=6 路 → 共 18 shard
#
# 必需:
#   TASK1_UDA_CKPT          checkpoint .pth
#   TASK1_UDA_PRED_OUT      输出目录（最终 pred/ 扁平；shards/ 为临时）
#   TASK1_UDA_CASES_JSON    splits JSON（取 train）
# 可选:
#   TASK1_UDA_NNUNET_RESULTS
#   TASK1_CUDA_VISIBLE_DEVICES / PRED_GPUS   默认 0,1,3
#   TASK1_UDA_PRED_PER_GPU                  默认 6
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK1_REPO_ROOT="${ROOT}"
export TASK1_BASE="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${TASK1_BASE}/task1_train_workspace}"
IMAGE_TAG="${IMAGE_TAG:-autopet_baseline}"

CKPT="${TASK1_UDA_CKPT:?need TASK1_UDA_CKPT}"
OUT="${TASK1_UDA_PRED_OUT:?need TASK1_UDA_PRED_OUT}"
CASES_JSON="${TASK1_UDA_CASES_JSON:?need TASK1_UDA_CASES_JSON}"
DATASET_ID="${DATASET_ID:-228}"
DS="Dataset${DATASET_ID}_AutoPETIV_Task1_2ch"
TRAINER="${TRAINER:-nnUNetTrainer_Task1StdTrainVal50}"
CONFIG="${CONFIG:-3d_fullres}"
_raw_fold="${FOLD:-0}"
if [[ "${_raw_fold}" =~ ^(all|[0-4])$ ]]; then
  FOLD="${_raw_fold}"
else
  FOLD=0
fi
RAW_IMG="${WORK}/nnUNet_raw/${DS}/imagesTr"
TRAINER_PY="${TASK1_TRAINER_PY:-${ROOT}/nnunet_ext_trainers/nnUNetTrainer_Task1StdTrainVal50.py}"
PER_GPU="${TASK1_UDA_PRED_PER_GPU:-6}"

[[ -f "${CKPT}" ]] || { echo "[error] missing ckpt: ${CKPT}" >&2; exit 1; }
[[ -f "${CASES_JSON}" ]] || { echo "[error] missing cases: ${CASES_JSON}" >&2; exit 1; }
[[ -d "${RAW_IMG}" ]] || { echo "[error] missing imagesTr: ${RAW_IMG}" >&2; exit 1; }

if [[ -n "${TASK1_UDA_NNUNET_RESULTS:-}" ]]; then
  NN_RESULTS="${TASK1_UDA_NNUNET_RESULTS}"
else
  _fold="$(dirname "${CKPT}")"
  _tf="$(dirname "${_fold}")"
  _ds="$(dirname "${_tf}")"
  NN_RESULTS="$(dirname "${_ds}")"
fi
[[ -d "${NN_RESULTS}/${DS}" ]] || {
  echo "[error] nnUNet_results 下无 ${DS}: ${NN_RESULTS}" >&2
  exit 1
}

# GPU 列表：PRED_GPUS 空格分隔，或从 CUDA_VISIBLE_DEVICES / DOCKER_GPUS 解析
if [[ -n "${PRED_GPUS:-}" ]]; then
  read -r -a GPU_ARR <<< "${PRED_GPUS}"
elif [[ -n "${TASK1_CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_ARR <<< "${TASK1_CUDA_VISIBLE_DEVICES}"
elif [[ "${TASK1_DOCKER_GPUS:-}" == device=* ]]; then
  IFS=',' read -r -a GPU_ARR <<< "${TASK1_DOCKER_GPUS#device=}"
else
  GPU_ARR=(0 1 3)
fi
N_GPU="${#GPU_ARR[@]}"
N_SHARDS=$(( N_GPU * PER_GPU ))
[[ "${N_SHARDS}" -ge 1 ]] || { echo "[error] N_SHARDS=0" >&2; exit 1; }

mkdir -p "${OUT}/pred" "${OUT}/shards"
chmod -R a+rwX "${OUT}" 2>/dev/null || true

_NNUNET_TRAINER_PKG="$(docker run --rm --entrypoint python3 "${IMAGE_TAG}" -c \
  'import nnunetv2,os; print(os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer"))')"

# 确保 fold 内有 ckpt 名（各 shard 共用同一 NN_RESULTS）
FOLD_DIR="$(find "${NN_RESULTS}/${DS}" -type d -path "*/fold_${FOLD}" | head -n1)"
[[ -n "${FOLD_DIR}" ]] || { echo "[error] fold dir not found" >&2; exit 1; }
CKPT_NAME="$(basename "${CKPT}")"
if [[ ! -f "${FOLD_DIR}/${CKPT_NAME}" ]]; then
  ln -sfn "${CKPT}" "${FOLD_DIR}/${CKPT_NAME}"
fi

echo "[uda-predict] gpus=${GPU_ARR[*]} per_gpu=${PER_GPU} n_shards=${N_SHARDS}"
echo "[uda-predict] results=${NN_RESULTS} ckpt=${CKPT} out=${OUT}/pred"
# 供空 loss 图表头 ETA 使用
date +%s >"${OUT}/predict_t0.txt"
chmod a+rw "${OUT}/predict_t0.txt" 2>/dev/null || true

# 建 shard：跳过已有完整 pred（nii+npz）的病例
export CASES_JSON RAW_IMG OUT N_SHARDS
python3 - <<'PY'
import json, os
from pathlib import Path

cases_json = Path(os.environ["CASES_JSON"])
raw = json.loads(cases_json.read_text(encoding="utf-8"))
if isinstance(raw, list) and raw and isinstance(raw[0], dict):
    cases = [str(x) for x in raw[0]["train"]]
elif isinstance(raw, dict) and "train" in raw:
    cases = [str(x) for x in raw["train"]]
else:
    cases = [str(x) for x in raw]

out = Path(os.environ["OUT"])
pred_flat = out / "pred"
pred_flat.mkdir(parents=True, exist_ok=True)
done = []
todo = []
def _prob_path(flat, case):
    # nnUNetv2 --save_probabilities → {case}.npz（偶见 .npz.npz）
    for name in (f"{case}.npz", f"{case}.npz.npz"):
        p = flat / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None

for c in cases:
    nii = pred_flat / f"{c}.nii.gz"
    npz = _prob_path(pred_flat, c)
    if nii.is_file() and npz is not None and nii.stat().st_size > 0:
        done.append(c)
    else:
        todo.append(c)

n_shards = int(os.environ["N_SHARDS"])
raw_img = Path(os.environ["RAW_IMG"])
shard_root = out / "shards"
# 清理旧 shard 输入（保留其他 round 无关）
for old in shard_root.glob("shard_*"):
    # 只清 imagesTs 链接，pred 可弃
    pass

(out / "cases.txt").write_text("\n".join(cases) + "\n", encoding="utf-8")
(out / "cases_todo.txt").write_text("\n".join(todo) + "\n", encoding="utf-8")
print(f"[uda-predict] total={len(cases)} done={len(done)} todo={len(todo)} shards={n_shards}")

missing = []
for i in range(n_shards):
    sd = shard_root / f"shard_{i}"
    inp = sd / "imagesTs"
    outp = sd / "pred"
    # 重建输入目录
    if inp.exists():
        for p in inp.glob("*"):
            p.unlink()
    inp.mkdir(parents=True, exist_ok=True)
    outp.mkdir(parents=True, exist_ok=True)
    sub = todo[i::n_shards]
    (sd / "cases.txt").write_text("\n".join(sub) + ("\n" if sub else ""), encoding="utf-8")
    for cid in sub:
        for ch in ("0000", "0001"):
            src = raw_img / f"{cid}_{ch}.nii.gz"
            if not src.is_file():
                missing.append(str(src))
                continue
            dst = inp / f"{cid}_{ch}.nii.gz"
            if not dst.exists():
                dst.symlink_to(src)
    print(f"[uda-predict] shard_{i} n={len(sub)}")

if missing:
    print("missing sample:", missing[:5])
    raise SystemExit(2)
PY
chmod -R a+rwX "${OUT}/shards" "${OUT}/pred" 2>/dev/null || true

n_todo=0
if [[ -f "${OUT}/cases_todo.txt" ]]; then
  n_todo="$(grep -c . "${OUT}/cases_todo.txt" || true)"
fi
n_todo="${n_todo:-0}"
if [[ "${n_todo}" -eq 0 ]]; then
  echo "[uda-predict] all cases already predicted → skip"
else
  declare -a PIDS=()
  declare -a SHARD_IDS=()
  for i in $(seq 0 $((N_SHARDS - 1))); do
    gpu="${GPU_ARR[$((i % N_GPU))]}"
    sd="${OUT}/shards/shard_${i}"
    ncase="$(wc -l <"${sd}/cases.txt" | tr -d ' ')"
    [[ "${ncase}" -gt 0 ]] || continue
    log="${sd}/run.log"
    echo "[uda-predict] launch shard_${i} gpu=${gpu} n=${ncase}"
    (
      docker run --rm \
        --user algorithm \
        --gpus "device=${gpu}" \
        --shm-size "${PRED_SHM_SIZE:-8g}" \
        -e "nnUNet_raw=${WORK}/nnUNet_raw" \
        -e "nnUNet_preprocessed=${WORK}/nnUNet_preprocessed" \
        -e "nnUNet_results=${NN_RESULTS}" \
        -e "PYTHONPATH=/home/algorithm/.local/lib/python3.11/site-packages" \
        -e "HOME=/home/algorithm" \
        -e "OMP_NUM_THREADS=1" \
        -e "MKL_NUM_THREADS=1" \
        -e "nnUNet_compile=false" \
        -v "${TRAINER_PY}:${_NNUNET_TRAINER_PKG}/nnUNetTrainer_Task1StdTrainVal50.py:ro" \
        -v "${WORK}:${WORK}" \
        -v "${sd}:${sd}" \
        --entrypoint bash \
        "${IMAGE_TAG}" -lc "
          set -euo pipefail
          nnUNetv2_predict \
            -i '${sd}/imagesTs' \
            -o '${sd}/pred' \
            -d ${DATASET_ID} \
            -tr ${TRAINER} \
            -c ${CONFIG} \
            -f ${FOLD} \
            --disable_tta \
            -npp 1 -nps 1 \
            -chk ${CKPT_NAME} \
            -device cuda \
            --save_probabilities
        "
    ) >"${log}" 2>&1 &
    PIDS+=("$!")
    SHARD_IDS+=("${i}")
  done

  rc=0
  for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    sid="${SHARD_IDS[$idx]}"
    if ! wait "${pid}"; then
      echo "[error] shard_${sid} failed；见 ${OUT}/shards/shard_${sid}/run.log" >&2
      tail -n 40 "${OUT}/shards/shard_${sid}/run.log" >&2 || true
      rc=1
    else
      echo "[uda-predict] shard_${sid} ok"
    fi
  done

  # flatten even if a shard died so the next run can resume from OUT/pred
  export OUT
  python3 - <<'PY'
from pathlib import Path
import os
import shutil
out = Path(os.environ["OUT"]) / "pred"
out.mkdir(parents=True, exist_ok=True)
n = 0
for sd in sorted((Path(os.environ["OUT"]) / "shards").glob("shard_*")):
    pred = sd / "pred"
    if not pred.is_dir():
        continue
    for p in pred.iterdir():
        if p.suffix == ".json":
            continue
        if p.name in ("dataset.json", "plans.json", "predict_from_raw_data_args.json"):
            continue
        dst = out / p.name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        shutil.copy2(p, dst)
        n += 1
print(f"[uda-predict] flattened_new_files={n}")
PY
fi

n_pred="$(find "${OUT}/pred" -maxdepth 1 -name '*.nii.gz' | wc -l | tr -d ' ')"
# nnUNet 写出 {case}.npz；兼容旧名 .npz.npz
n_npz="$(find "${OUT}/pred" -maxdepth 1 \( -name '*.npz' -o -name '*.npz.npz' \) ! -name '*.pkl' | wc -l | tr -d ' ')"
n_cases="$(grep -c . "${OUT}/cases.txt" || true)"
echo "[uda-predict] done nii=${n_pred} npz=${n_npz} cases=${n_cases} → ${OUT}/pred"
[[ "${n_pred}" -ge 1 ]] || { echo "[error] no predictions" >&2; exit 1; }
# 允许略少（个别失败）但告警
if [[ "${n_pred}" -lt "${n_cases}" ]]; then
  echo "[warn] nii ${n_pred} < cases ${n_cases}" >&2
fi
[[ "${n_pred}" -ge $((n_cases * 9 / 10)) ]] || {
  echo "[error] too few preds: ${n_pred}/${n_cases}" >&2
  exit 1
}
