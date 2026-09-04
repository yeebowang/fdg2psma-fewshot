#!/usr/bin/env bash
# SegAnyPET fewshot50 on folds 2,5,8 — PET-only click finetune from seganypet_v2 / lesion.
# Parallel: f2→GPU0, f5→GPU1, f8→GPU3
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"
SEG_CODE="${CTRL}/ICLR2026/third_party/SegAnyPET/code"
SEG_PIP="${CTRL}/ICLR2026/third_party/seganypet_pip"
WEIGHT_DIR="${REPO}/weights/seganypet"

FEWSHOT_N="${TASK1_FEWSHOT_N:-50}"
BOARD_STAGE="${TASK1_PSMA_BOARD_STAGE:-psma_fs${FEWSHOT_N}_f258}"

FT_EPOCHS="${TASK1_SEGANY_EPOCHS:-100}"
BATCH_SIZE="${TASK1_SEGANY_BATCH_SIZE:-2}"
ACCUM="${TASK1_SEGANY_ACCUM:-4}"
FOLDS_CSV="${TASK1_SEGANY_FOLDS_CSV:-2,5,8}"
FT_GPU_LIST="${TASK1_SEGANY_GPU_LIST:-0 1 3}"
# 1 GPU / fold container → default 6 workers per GPU
WORKERS_PER_GPU="${TASK1_SEGANY_WORKERS_PER_GPU:-6}"
WORKERS="${TASK1_SEGANY_WORKERS:-${WORKERS_PER_GPU}}"
VAL_INTERVAL="${TASK1_SEGANY_VAL_INTERVAL:-20}"
VAL_CLICKS="${TASK1_SEGANY_VAL_CLICKS:-5}"
VAL_MAX_CASES="${TASK1_SEGANY_VAL_MAX_CASES:-15}"
LR_MODE="${TASK1_SEGANY_LR_MODE:-finetune}"
LR="${TASK1_SEGANY_LR:-8e-4}"
MILESTONES="${TASK1_SEGANY_MILESTONES:-60,85}"
CLICK_MAX="${TASK1_SEGANY_CLICK_MAX:-11}"
EXTRA_ARGS=()
if [[ "${LR_MODE}" == "official" ]]; then
  EXTRA_ARGS+=(--lr-mode official --lr "${LR}" --milestones "${MILESTONES}" --click-max "${CLICK_MAX}")
fi
# single-GPU parallel: never DataParallel across leftover devices
EXTRA_ARGS+=(--no-dataparallel)

BOARD_METHOD="${TASK1_BOARD_METHOD:-seganypet}"
# prefer lesion (tumor-centric); fall back to v2. Scratch fewshot uses FDG ckpt via TASK1_SEGANY_CKPT.
if [[ -n "${TASK1_SEGANY_CKPT:-}" ]]; then
  CKPT="${TASK1_SEGANY_CKPT}"
elif [[ "${BOARD_METHOD}" == "seganypet_scratch" ]]; then
  echo "[error] seganypet_scratch fewshot needs TASK1_SEGANY_CKPT (FDG scratch best.pth)" >&2
  exit 1
elif [[ -f "${WEIGHT_DIR}/seganypet_lesion.pth" ]]; then
  CKPT="${WEIGHT_DIR}/seganypet_lesion.pth"
elif [[ -f "${WEIGHT_DIR}/seganypet_v2.pth" ]]; then
  CKPT="${WEIGHT_DIR}/seganypet_v2.pth"
else
  echo "[error] missing SegAnyPET weights under ${WEIGHT_DIR}" >&2
  ls -lh "${WEIGHT_DIR}" >&2 || true
  exit 1
fi

DATA_ROOT="${TASK1_SEGANY_DATA_ROOT:-${DATA}/task1_train_workspace/seganypet_fewshot${FEWSHOT_N}_f258}"
SPLIT_DIR="${TASK1_FEWSHOT_SPLIT_DIR:-${CTRL}/ICLR2026/data/splits_mae_psma_fewshot${FEWSHOT_N}_9fold}"
LOG_DIR="${CTRL}/ICLR2026/vis"

STAMP_TZ="${TASK1_STAMP_TZ:-Asia/Shanghai}"
if [[ -n "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME}"
elif [[ "${BOARD_METHOD}" == "seganypet_scratch" ]]; then
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_seganypet_scratch_psma_fs${FEWSHOT_N}_f258_gpu013"
else
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_seganypet_fs${FEWSHOT_N}_f258_gpu013"
fi

OUT_ROOT="${REPO}/runs/${STAMP}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_BASE="${DATA}"

echo "[seganypet-fs${FEWSHOT_N}] STAMP=${STAMP}"
echo "[seganypet-fs${FEWSHOT_N}] ckpt=${CKPT}"
echo "[seganypet-fs${FEWSHOT_N}] data=${DATA_ROOT} ep=${FT_EPOCHS} bs=${BATCH_SIZE} accum=${ACCUM} lr_mode=${LR_MODE} folds=${FOLDS_CSV}"

[[ -d "${SEG_PIP}/torchio" ]] || {
  echo "[error] torchio not in ${SEG_PIP}; install first" >&2
  exit 1
}

# prepare PET-only fold folders
python3 "${CTRL}/ICLR2026/scripts/prepare_seganypet_fewshot_f258.py" \
  --folds "${FOLDS_CSV}" \
  --split-dir "${SPLIT_DIR}" \
  --out-root "${DATA_ROOT}"

bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

IFS=',' read -r -a FOLD_ARR <<< "${FOLDS_CSV}"
read -r -a GPU_ARR <<< "${FT_GPU_LIST}"
[[ "${#GPU_ARR[@]}" -ge 1 ]] || { echo "[error] empty GPU list" >&2; exit 1; }

STATUS="${OUT_ROOT}/status.jsonl"
echo "{\"event\":\"start\",\"stamp\":\"${STAMP}\",\"ckpt\":\"${CKPT}\",\"folds\":\"${FOLDS_CSV}\"}" >>"${STATUS}"

BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"${BOARD_STAGE}\":{\"status\":\"running\",\"stamp\":\"${STAMP}\",\"init\":\"${CKPT}\"}}},\"updated_note\":\"${BOARD_METHOD} fs${FEWSHOT_N} start\"}" || true

run_one_bg() {
  local fold="$1" gpu="$2"
  local fold_data="${DATA_ROOT}/fold${fold}"
  local out_dir="${OUT_ROOT}/seganypet/fold${fold}"
  local log="${LOG_DIR}/nohup_seganypet_fs${FEWSHOT_N}_fold${fold}_${STAMP}.log"
  local cname="seganypet_fs${FEWSHOT_N}_f${fold}_${STAMP}"
  local fresh_flag=(--fresh)
  if [[ "${TASK1_SEGANY_FORCE_FRESH:-0}" != "1" ]] && [[ -f "${out_dir}/latest.pth" ]]; then
    fresh_flag=()
    echo "[seganypet-fs${FEWSHOT_N}] resume f${fold}"
  fi
  mkdir -p "${out_dir}"
  docker rm -f "${cname}" >/dev/null 2>&1 || true
  echo "[seganypet-fs${FEWSHOT_N}] === fold${fold} → GPU${gpu} ==="
  nohup docker run --rm \
    --name "${cname}" \
    --gpus "device=${gpu}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONPATH="${SEG_PIP}:${SEG_CODE}:${CTRL}/ICLR2026/scripts" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "${CTRL}:${CTRL}" \
    -v "${DATA}:${DATA}" \
    -w "${SEG_CODE}" \
    --shm-size=8g \
    "${IMAGE}" \
    python3 "${CTRL}/ICLR2026/scripts/seganypet_fewshot_finetune.py" \
      --data-root "${fold_data}" \
      --checkpoint "${CKPT}" \
      --out-dir "${out_dir}" \
      --epochs "${FT_EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --accumulation-steps "${ACCUM}" \
      --num-workers "${WORKERS}" \
      --val-interval "${VAL_INTERVAL}" \
      --val-clicks "${VAL_CLICKS}" \
      --val-max-cases "${VAL_MAX_CASES}" \
      "${EXTRA_ARGS[@]}" \
      "${fresh_flag[@]}" \
    >"${log}" 2>&1 &
  echo $! >"${out_dir}/docker_host_pid.txt"
}

t0=$(date +%s)
any_fail=0
n_gpu="${#GPU_ARR[@]}"
idx=0
n_fold="${#FOLD_ARR[@]}"
while [[ "${idx}" -lt "${n_fold}" ]]; do
  wave_end=$((idx + n_gpu))
  [[ "${wave_end}" -gt "${n_fold}" ]] && wave_end="${n_fold}"
  echo "[seganypet-fs${FEWSHOT_N}] wave folds ${FOLD_ARR[*]:idx:$((wave_end - idx))} on GPUs ${GPU_ARR[*]}"
  for ((i = idx; i < wave_end; i++)); do
    g=$((i - idx))
    run_one_bg "${FOLD_ARR[$i]}" "${GPU_ARR[$g]}"
  done
  for ((i = idx; i < wave_end; i++)); do
    fold="${FOLD_ARR[$i]}"
    out_dir="${OUT_ROOT}/seganypet/fold${fold}"
    pid="$(cat "${out_dir}/docker_host_pid.txt")"
    set +e
    wait "${pid}"
    rc=$?
    set -e
    echo "{\"event\":\"fold_done\",\"fold\":${fold},\"rc\":${rc}}" >>"${STATUS}"
    if [[ "${rc}" -ne 0 ]]; then
      echo "[error] seganypet f${fold} rc=${rc} log=${LOG_DIR}/nohup_seganypet_fs${FEWSHOT_N}_fold${fold}_${STAMP}.log" >&2
      any_fail=1
    fi
  done
  idx="${wave_end}"
done
t1=$(date +%s)
echo "{\"event\":\"wave_done\",\"sec\":$((t1 - t0))}" >>"${STATUS}"
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true
[[ "${any_fail}" -eq 0 ]] || exit 1

python3 - <<PY
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
folds = [int(x) for x in "${FOLDS_CSV}".split(",") if x.strip()]
dices = []
for fold in folds:
    best = None
    m = root / "seganypet" / f"fold{fold}" / "metrics.jsonl"
    if m.is_file():
        for line in m.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            vd = r.get("val_dice")
            if vd is not None and vd == vd:
                best = float(vd) if best is None or float(vd) > best else best
    s = root / "seganypet" / f"fold{fold}" / "summary.json"
    if s.is_file():
        bj = json.loads(s.read_text()).get("best_val_dice")
        if bj is not None and (best is None or bj > best):
            best = float(bj)
    dices.append(best)
ok = [d for d in dices if d is not None]
mean = sum(ok) / len(ok) if ok else None
std = (sum((x - mean) ** 2 for x in ok) / len(ok)) ** 0.5 if ok and mean is not None else None
summary = {
    "stamp": "${STAMP}",
    "folds": folds,
    "protocol": "fewshot${FEWSHOT_N}_seganypet_click",
    "checkpoint": "${CKPT}",
    "note": "val Dice with GT-derived clicks (SegAnyPET official prompt protocol); not click-free auto-seg",
    "fold_best_dice": dices,
    "mean": mean,
    "std": std,
    "n_ok": len(ok),
}
out = root / "aggregate_val_dice_f258.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
vis = Path("${LOG_DIR}") / f"aggregate_seganypet_fs${FEWSHOT_N}_f258_${STAMP}.json"
vis.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" --ingest-seganypet-stamp "${STAMP}" || true
echo "[seganypet-fs${FEWSHOT_N}] ALL DONE STAMP=${STAMP}"
echo "STAMP=${STAMP}" > "${LOG_DIR}/iclr2026_seganypet_f258_${STAMP}.txt"
