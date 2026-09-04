#!/usr/bin/env bash
# Final TEST20 eval for MAE/MONAI fewshot f258.
# Layout: 各站(折)一卡并行 · f2→0 / f5→1 / f8→3（与 PSMA 训练一致）
#
#   METHOD=mae|monai STAMP=<run_stamp> bash ICLR2026/run/run_eval_psma_test20_f258_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"

FEWSHOT_N="${TASK1_FEWSHOT_N:-50}"
BOARD_STAGE="${TASK1_PSMA_BOARD_STAGE:-psma_fs${FEWSHOT_N}_f258}"

METHOD="${METHOD:-mae}"          # mae | monai
STAMP="${STAMP:?set STAMP}"
FOLDS_CSV="${TASK1_MAE_FEWSHOT_FOLDS_CSV:-2,5,8}"
FOLD_GPUS_CSV="${TASK1_FOLD_GPUS:-2:0,5:1,8:3}"
TEST_JSON="${CTRL}/ICLR2026/data/splits_mae_psma_test20.json"
CACHE="${TASK1_MAE_PSMA_CACHE_DIR:-${DATA}/task1_train_workspace/mae_cache/psma_baseline2_70_10}"
RAW_IMG="${TASK1_MAE_IMAGES_TR:-${DATA}/dataset1/imagesTr}"
RAW_LAB="${TASK1_MAE_LABELS_TR:-${DATA}/dataset1/labelsTr}"
SKIP_DONE="${TASK1_TEST_SKIP_DONE:-1}"

if [[ "${METHOD}" == "mae" ]]; then
  SUB=mae
  STEM_PREFIX=seg_psma_fs${FEWSHOT_N}_fdgseg_f
  DEPTHS=2,2,6,2
  USE_V2=1
  BOARD_KEY=mae_swinunetr
elif [[ "${METHOD}" == "mae_scratch" ]]; then
  SUB=mae
  STEM_PREFIX=seg_psma_fs${FEWSHOT_N}_fdgseg_f
  DEPTHS=2,2,6,2
  USE_V2=1
  BOARD_KEY=mae_scratch
elif [[ "${METHOD}" == "monai" ]]; then
  SUB=monai
  STEM_PREFIX=seg_psma_fs${FEWSHOT_N}_monai_fdgseg_f
  DEPTHS=2,2,2,2
  USE_V2=0
  BOARD_KEY=monai_swinvit
elif [[ "${METHOD}" == "monai_scratch" ]]; then
  SUB=monai
  STEM_PREFIX=seg_psma_fs${FEWSHOT_N}_monai_fdgseg_f
  DEPTHS=2,2,2,2
  USE_V2=0
  BOARD_KEY=monai_scratch
else
  echo "[error] METHOD=mae|mae_scratch|monai|monai_scratch" >&2
  exit 1
fi

OUT_ROOT="${REPO}/runs/${STAMP}"
EVAL_ROOT="${OUT_ROOT}/psma_test20_eval"
mkdir -p "${EVAL_ROOT}" "${CACHE}" "${LOG_DIR}"

declare -A GPU_OF
IFS=',' read -r -a _pairs <<< "${FOLD_GPUS_CSV}"
for p in "${_pairs[@]}"; do
  GPU_OF["${p%%:*}"]="${p##*:}"
done
IFS=',' read -r -a FOLD_ARR <<< "${FOLDS_CSV}"

echo "[test20] METHOD=${METHOD} STAMP=${STAMP} layout=1gpu/fold map=${FOLD_GPUS_CSV}"

# ensure test20 cache (once)
docker run --rm \
  -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
  "${IMAGE}" \
  python3 "${CTRL}/ICLR2026/scripts/mae_preprocess_fdg_cache.py" \
    --cases-json "${TEST_JSON}" \
    --images-tr "${RAW_IMG}" \
    --labels-tr "${RAW_LAB}" \
    --out-dir "${CACHE}" \
    --workers "${TASK1_MAE_PREP_WORKERS:-16}"

_run_fold() {
  local fold="$1" gpu="$2"
  local out_dir="${OUT_ROOT}/${SUB}/fold${fold}"
  local ckpt_best="${out_dir}/best_${STEM_PREFIX}${fold}.pth"
  local ckpt_latest="${out_dir}/latest_${STEM_PREFIX}${fold}.pth"
  local ckpt=""
  if [[ "${TASK1_TEST_CKPT:-best}" == "latest" ]]; then
    [[ -f "${ckpt_latest}" ]] && ckpt="${ckpt_latest}"
    [[ -z "${ckpt}" && -f "${ckpt_best}" ]] && ckpt="${ckpt_best}"
  else
    [[ -f "${ckpt_best}" ]] && ckpt="${ckpt_best}"
    [[ -z "${ckpt}" && -f "${ckpt_latest}" ]] && ckpt="${ckpt_latest}"
  fi
  [[ -n "${ckpt}" && -f "${ckpt}" ]] || { echo "[error] missing ckpt fold${fold} mode=${TASK1_TEST_CKPT:-best}" >&2; return 1; }
  local out_json="${EVAL_ROOT}/fold${fold}_test20.json"
  local log="${LOG_DIR}/nohup_${METHOD}_test20_fold${fold}_${STAMP}.log"

  if [[ "${SKIP_DONE}" == "1" && -f "${out_json}" ]]; then
    echo "[test20] skip fold${fold} (json exists)"
    return 0
  fi

  echo "[test20] fold${fold} → GPU${gpu} ckpt=${ckpt}"
  docker run --rm \
    --gpus "device=${gpu}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    -w "${REPO}" --shm-size=16g \
    "${IMAGE}" \
    python3 "${CTRL}/ICLR2026/scripts/mae_eval_seg_psma_test.py" \
      --cases-json "${TEST_JSON}" \
      --cache-dir "${CACHE}" \
      --seg-ckpt "${ckpt}" \
      --out-json "${out_json}" \
      --depths "${DEPTHS}" \
      --use-v2 "${USE_V2}" \
      --tag "${METHOD}_f${fold}_${STAMP}" \
    >"${log}" 2>&1
}

pids=()
wave_n=0
rc=0
for fold in "${FOLD_ARR[@]}"; do
  gpu="${GPU_OF[${fold}]:-}"
  if [[ -z "${gpu}" ]]; then
    # cycle default GPUs 0,1,3 when map is incomplete (9-fold)
    case $((wave_n % 3)) in
      0) gpu=0 ;;
      1) gpu=1 ;;
      *) gpu=3 ;;
    esac
  fi
  echo "[test20] queue fold${fold} → GPU${gpu}"
  _run_fold "${fold}" "${gpu}" &
  pids+=($!)
  wave_n=$((wave_n + 1))
  # at most 3 concurrent evals (one per GPU)
  if [[ "${#pids[@]}" -ge 3 ]]; then
    wait "${pids[0]}" || rc=1
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do
  wait "${pid}" || rc=1
done
[[ "${rc}" -eq 0 ]] || { echo "[error] some TEST folds failed" >&2; exit 1; }

python3 - <<PY
import json, statistics
from pathlib import Path
folds = [int(x) for x in "${FOLDS_CSV}".split(",") if x.strip()]
root = Path("${EVAL_ROOT}")
fd, vals = {}, []
fold_scores = []
for f in folds:
    p = root / f"fold{f}_test20.json"
    if not p.is_file():
        raise SystemExit(f"missing {p}")
    d = json.loads(p.read_text())
    md = d.get("mean_dice_positive", d.get("mean_dice"))
    if md is not None and md == md:
        fd[str(f)] = float(md)
        vals.append(float(md))
    fold_scores.append(d)
mean = sum(vals)/len(vals) if vals else None
std = statistics.pstdev(vals) if len(vals) > 1 else 0.0

def _micro(key_num, key_den, rate_key):
    nums, dens, rates = [], [], []
    for d in fold_scores:
        n, den = d.get(key_num), d.get(key_den)
        if isinstance(n, (int, float)) and isinstance(den, (int, float)) and den:
            nums.append(float(n)); dens.append(float(den))
        r = d.get(rate_key, d.get("mean_" + rate_key.split("_")[0] if False else rate_key))
        r = d.get(rate_key)
        if r is None:
            r = d.get("mean_fp" if rate_key == "fp_rate" else "mean_fn")
        if isinstance(r, (int, float)) and r == r:
            rates.append(float(r))
    if dens and sum(dens) > 0:
        return sum(nums) / sum(dens)
    return (sum(rates) / len(rates)) if rates else None

fp = _micro("sum_fp", "sum_neg_voxels", "fp_rate")
fn = _micro("sum_fn", "sum_pos_voxels", "fn_rate")
val_agg = Path("${OUT_ROOT}") / "aggregate_val_dice_f258.json"
val_mean, val_fd = None, {}
if val_agg.is_file():
    va = json.loads(val_agg.read_text())
    val_mean = va.get("mean")
    vfd = va.get("fold_best_dice") or {}
    if isinstance(vfd, dict):
        val_fd = {str(k): v for k, v in vfd.items()}
summary = {
    "stamp": "${STAMP}",
    "method": "${METHOD}",
    "split": "PSMA_TEST20",
    "n_test": 120,
    "test_layout": "1gpu_per_fold_parallel",
    "fold_gpus": "${FOLD_GPUS_CSV}",
    "protocol": "online_val=PSMA_VAL10; final=PSMA_TEST20",
    "fold_test_dice": fd,
    "test_mean": mean,
    "test_std": std,
    "fp_rate": fp,
    "fn_rate": fn,
    "mean_fp": fp,
    "mean_fn": fn,
    "val_monitor_mean": val_mean,
    "val_monitor_fold_dice": val_fd,
}
out = root / "aggregate_test20_f258.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
vis = Path("${LOG_DIR}") / f"aggregate_${METHOD}_psma_test20_f258_${STAMP}.json"
vis.write_text(json.dumps(summary, indent=2) + "\n")
board_p = Path("${BOARD_JSON}")
if board_p.is_file():
    board = json.loads(board_p.read_text())
    st = board["methods"]["${BOARD_KEY}"]["${BOARD_STAGE}"]
    st["status"] = "done"
    st["stamp"] = "${STAMP}"
    old_fd = st.get("fold_dice") or {}
    merged = {str(k): v for k, v in old_fd.items() if isinstance(v, (int, float))} if isinstance(old_fd, dict) else {}
    merged.update({str(k): v for k, v in fd.items()})
    st["fold_dice"] = merged
    mvals = [v for v in merged.values() if isinstance(v, (int, float))]
    st["mean"] = (sum(mvals) / len(mvals)) if mvals else mean
    st["eval_done"] = len(mvals)
    st["eval_total"] = 9
    if isinstance(fp, (int, float)) and fp == fp:
        st["mean_fp"] = float(fp)
    if isinstance(fn, (int, float)) and fn == fn:
        st["mean_fn"] = float(fn)
    st["val_monitor_mean"] = val_mean
    st["val_monitor_fold_dice"] = val_fd
    st["metric"] = "TEST20 Dice/FP/FN (final); VAL10=monitor"
    st["note"] = (
        f"TEST20 DONE · {100*mean:.2f}%/"
        f"{(f'{100*fp:.2f}%' if isinstance(fp,(int,float)) and fp==fp else '—')}/"
        f"{(f'{100*fn:.2f}%' if isinstance(fn,(int,float)) and fn==fn else '—')}"
        if mean is not None else st.get("note")
    )
    board["updated_note"] = f"${METHOD} TEST20 mean={mean} FP={fp} FN={fn}"
    board_p.write_text(json.dumps(board, indent=2) + "\n")
print(json.dumps({k: summary[k] for k in summary if k != "val_monitor_fold_dice"}, indent=2))
PY

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD_JSON}" || true
echo "[test20] DONE ${METHOD} ${STAMP}"
