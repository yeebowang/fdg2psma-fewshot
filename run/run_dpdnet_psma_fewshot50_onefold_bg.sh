#!/usr/bin/env bash
# DpDNet PSMA fewshot50 · single fold · 1 GPU · MAE full-case:
#   tr=floor(n_train/bs) · val=ceil(n_val/bs) · val every 20ep · 100ep
#   default bs=2 → tr=25 · val=30 (n=50/59)
# Required: FOLD_ID (2/5/8), GPU_ID, PARENT_STAMP
# Optional: TASK1_DPDNET_FDG_BEST / TASK1_DPDNET_FDG_STAMP
# Resume: --c only (no -pretrained_weights) when fold ckpt exists.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
ICLR_VIS="${CTRL}/ICLR2026/vis"
DPD="${CTRL}/ICLR2026/third_party/DpDNet"
IMAGE="${TASK1_NNUNET_IMAGE:-autopet_baseline:latest}"
mkdir -p "${ICLR_VIS}"

FOLD_ID="${FOLD_ID:?need FOLD_ID}"
GPU_ID="${GPU_ID:?need GPU_ID}"
DATASET_ID="${DATASET_ID:-240}"
DS="Dataset${DATASET_ID}_DpDNet_PSMA_2ch"
TRAINER="${TRAINER:-STUNetTrainer_small_prompt}"
CONFIG="${CONFIG:-3d_fullres}"
TF="${TRAINER}__nnUNetPlans__${CONFIG}"
TOTAL_EPOCHS="${TASK1_DPDNET_NUM_EPOCHS:-${TASK1_NUM_EPOCHS:-100}}"
TRAIN_ITERS="${TASK1_DPDNET_TRAIN_ITERS:-${TASK1_TRAIN_ITERS_PER_EPOCH:-25}}"
# MAE full-case default (bs=2, n_train=50, n_val=59): tr=25, val=30
VAL_ITERS="${TASK1_DPDNET_VAL_ITERS:-${TASK1_FS50_VAL_ITERS:-${TASK1_VAL_ITERS_PER_EPOCH:-25}}}"
VAL_EVERY="${TASK1_DPDNET_VAL_EVERY:-${TASK1_FS50_VAL_EVERY_N_EPOCHS:-${TASK1_VAL_EVERY_N_EPOCHS:-20}}}"
BATCH="${TASK1_DPDNET_BATCH_SIZE:-${TASK1_FIXED_BATCH_3D_FULLRES:-2}}"
N_PROC_DA="${TASK1_DPDNET_N_PROC_DA:-${TASK1_N_PROC_DA:-4}}"
BEST_BY="${TASK1_BEST_BY:-ema_fg_dice}"

FDG_DS="Dataset239_DpDNet_FDG_2ch"
FDG_TF="${TASK1_DPDNET_FDG_TF:-${TRAINER}__nnUNetPlans__3d_fullres}"
# Official aligned FDG: dpdnet_fdg_LAST_STAMP.txt wins over stale caller env.
FDG_STAMP=""
FDG_BEST=""
if [[ "${TASK1_DPDNET_FDG_FORCE_STAMP:-0}" != "1" && -f "${ICLR_VIS}/dpdnet_fdg_LAST_STAMP.txt" ]]; then
  FDG_STAMP="$(tr -d '[:space:]' < "${ICLR_VIS}/dpdnet_fdg_LAST_STAMP.txt")"
elif [[ -n "${TASK1_DPDNET_FDG_STAMP:-}" ]]; then
  FDG_STAMP="${TASK1_DPDNET_FDG_STAMP}"
fi
if [[ "${TASK1_DPDNET_FDG_FORCE_STAMP:-0}" == "1" ]]; then
  FDG_STAMP="${TASK1_DPDNET_FDG_STAMP:-}"
  FDG_BEST="${TASK1_DPDNET_FDG_BEST:-}"
fi
if [[ -z "${FDG_BEST}" && -n "${FDG_STAMP}" ]]; then
  _fold="${WORK}/nnUNet_results/${FDG_STAMP}/${FDG_DS}/${FDG_TF}/fold_0"
  for _c in checkpoint_final.pth checkpoint_latest.pth checkpoint_best.pth; do
    if [[ -f "${_fold}/${_c}" ]]; then
      FDG_BEST="${_fold}/${_c}"
      break
    fi
  done
fi
if [[ -n "${TASK1_DPDNET_FDG_STAMP:-}" && "${TASK1_DPDNET_FDG_STAMP}" != "${FDG_STAMP}" ]]; then
  echo "[dpdnet-psma-1f] FDG override: caller=${TASK1_DPDNET_FDG_STAMP} -> official=${FDG_STAMP} (${FDG_BEST})"
fi
[[ -n "${FDG_BEST}" && -f "${FDG_BEST}" ]] || {
  echo "[error] missing FDG final/latest ckpt: ${FDG_BEST:-<unset>} (set TASK1_DPDNET_FDG_BEST)" >&2
  exit 1
}

PARENT="${PARENT_STAMP:-${TASK1_NNUNET_RESULTS_STAMP_NAME:-}}"
[[ -n "${PARENT}" ]] || {
  PARENT="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_dpdnet_psma_fs50_f${FOLD_ID}_1gpu_bs${BATCH}_tr${TRAIN_ITERS}_val${VAL_ITERS}e${VAL_EVERY}_${TOTAL_EPOCHS}ep"
}
if [[ "${PARENT}" == *_f${FOLD_ID} ]]; then
  STAMP="${PARENT}"
  PARENT="${PARENT%_f${FOLD_ID}}"
else
  STAMP="${PARENT}_f${FOLD_ID}"
fi
export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"

RESULTS_ROOT="${WORK}/nnUNet_results/${STAMP}"
mkdir -p "${RESULTS_ROOT}"
docker run --rm --user root -v "${DATA}:${DATA}" --entrypoint bash "${IMAGE}" -lc \
  "chmod -R a+rwX '${RESULTS_ROOT}' 2>/dev/null || true" || true

LOG="${ICLR_VIS}/nohup_dpdnet_psma_fs50_fold${FOLD_ID}_${STAMP}.log"
CNAME="dpdnet_psma_f${FOLD_ID}_${STAMP}"

echo "[dpdnet-psma-1f] STAMP=${STAMP} fold=${FOLD_ID} gpu=${GPU_ID} bs=${BATCH} ep=${TOTAL_EPOCHS} tr=${TRAIN_ITERS} val=${VAL_ITERS} every=${VAL_EVERY} best=${BEST_BY}"
echo "[dpdnet-psma-1f] pretrained=${FDG_BEST}"

export TASK1_PREFLIGHT_GPUS="${GPU_ID}"
export TASK1_PREFLIGHT_LABEL="iclr2026-dpdnet-psma-f${FOLD_ID}"
bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true
TASK1_CRASH_MONITOR_STAGE="dpdnet_psma_f${FOLD_ID}_start" \
TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" \
  bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

# plans batch_size (shared Dataset240)
python3 - <<PY
import json
from pathlib import Path
p = Path("${WORK}/nnUNet_preprocessed/${DS}/nnUNetPlans.json")
d = json.loads(p.read_text())
cfg = d["configurations"]["3d_fullres"]
old = cfg.get("batch_size")
cfg["batch_size"] = int("${BATCH}")
p.write_text(json.dumps(d, indent=2) + "\n")
print(f"[dpdnet-psma-1f] plans batch_size {old} -> ${BATCH}")
PY

# guard
export TASK1_GUARD_STAMP="${STAMP}"
export TASK1_GUARD_TRAINER_FOLDER="${TF}"
export TASK1_GUARD_DATASET_DIR="${DS}"
export TASK1_GUARD_TOTAL_EPOCHS="${TOTAL_EPOCHS}"
export TASK1_GUARD_RESTART_SCRIPT="ICLR2026/run/run_dpdnet_psma_fewshot50_onefold_bg.sh"
export TASK1_GUARD_REQUIRE_ARM=1
export FOLD="${FOLD_ID}"
export TASK1_GUARD_EXTRA_ENV="FOLD_ID=${FOLD_ID},GPU_ID=${GPU_ID},PARENT_STAMP=${PARENT},TASK1_DPDNET_SKIP_PREPARE=1,TASK1_DPDNET_BATCH_SIZE=${BATCH},TASK1_DPDNET_NUM_EPOCHS=${TOTAL_EPOCHS},TASK1_DPDNET_TRAIN_ITERS=${TRAIN_ITERS},TASK1_DPDNET_VAL_ITERS=${VAL_ITERS},TASK1_DPDNET_VAL_EVERY=${VAL_EVERY},TASK1_VAL_EVERY_N_EPOCHS=${VAL_EVERY},TASK1_BEST_BY=${BEST_BY},TASK1_DPDNET_N_PROC_DA=${N_PROC_DA},TASK1_DPDNET_FDG_BEST=${FDG_BEST},TASK1_NNUNET_RESULTS_STAMP_NAME=${STAMP}"
bash "${CTRL}/run_task/run_task1_train_auto_resume_guard_bg.sh" || true

docker rm -f "${CNAME}" >/dev/null 2>&1 || true
rm -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${STAMP}.txt"

FOLD_DIR="${RESULTS_ROOT}/${DS}/${TF}/fold_${FOLD_ID}"
# nnUNetv2 uses --c (not -c). Continue forbids -pretrained_weights at the same time.
TRAIN_TAIL="-pretrained_weights '${FDG_BEST}'"
if [[ -f "${FOLD_DIR}/checkpoint_latest.pth" || -f "${FOLD_DIR}/checkpoint_best.pth" || -f "${FOLD_DIR}/checkpoint_final.pth" ]]; then
  TRAIN_TAIL="--c"
  echo "[dpdnet-psma-1f] resume --c (found ckpt in ${FOLD_DIR}; skip pretrained)"
fi

nohup docker run --rm \
  --name "${CNAME}" \
  --gpus "\"device=${GPU_ID}\"" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HOME=/home/algorithm \
  -e TASK1_DPDNET_NUM_EPOCHS="${TOTAL_EPOCHS}" \
  -e TASK1_DPDNET_TRAIN_ITERS="${TRAIN_ITERS}" \
  -e TASK1_DPDNET_VAL_ITERS="${VAL_ITERS}" \
  -e TASK1_TRAIN_ITERS_PER_EPOCH="${TRAIN_ITERS}" \
  -e TASK1_VAL_ITERS_PER_EPOCH="${VAL_ITERS}" \
  -e TASK1_VAL_EVERY_N_EPOCHS="${VAL_EVERY}" \
  -e TASK1_DPDNET_VAL_EVERY="${VAL_EVERY}" \
  -e TASK1_BEST_BY="${BEST_BY}" \
  -e nnUNet_raw="${WORK}/nnUNet_raw" \
  -e nnUNet_preprocessed="${WORK}/nnUNet_preprocessed" \
  -e nnUNet_results="${RESULTS_ROOT}" \
  -e PYTHONPATH="${DPD}:/home/algorithm/.local/lib/python3.11/site-packages" \
  -e nnUNet_n_proc_DA="${N_PROC_DA}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TASK1_DPDNET_CT_ENCODER="${TASK1_DPDNET_CT_ENCODER:-}" \
  -e TASK1_DPDNET_PET_ENCODER="${TASK1_DPDNET_PET_ENCODER:-}" \
  -e TASK1_DPDNET_SKIP_ENCODER_INIT="${TASK1_DPDNET_SKIP_ENCODER_INIT:-0}" \
  -v "${CTRL}:${CTRL}" \
  -v "${DATA}:${DATA}" \
  --shm-size=16g \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "mkdir -p '${RESULTS_ROOT}' && nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD_ID} -tr ${TRAINER} ${TRAIN_TAIL}" \
  >"${LOG}" 2>&1 &
echo $! > "${RESULTS_ROOT}/nohup.pid"
sleep 12

TASK1_CRASH_MONITOR_STAGE="dpdnet_psma_f${FOLD_ID}_running" \
TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" \
TASK1_CRASH_MONITOR_ARM_SEC="${TASK1_CRASH_MONITOR_ARM_SEC:-86400}" \
  bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true

echo "[dpdnet-psma-1f] launched STAMP=${STAMP} log=${LOG} cname=${CNAME}"
echo "STAMP=${STAMP}" > "${ICLR_VIS}/iclr2026_dpdnet_psma_fs50_f${FOLD_ID}_${STAMP}.txt"
echo "LOG=${LOG}" >> "${ICLR_VIS}/iclr2026_dpdnet_psma_fs50_f${FOLD_ID}_${STAMP}.txt"
echo "FOLD=${FOLD_ID} GPU=${GPU_ID} BS=${BATCH} TR=${TRAIN_ITERS} VAL=${VAL_ITERS}e${VAL_EVERY} EP=${TOTAL_EPOCHS} BEST=${BEST_BY}" >> "${ICLR_VIS}/iclr2026_dpdnet_psma_fs50_f${FOLD_ID}_${STAMP}.txt"
echo "FDG_BEST=${FDG_BEST}" >> "${ICLR_VIS}/iclr2026_dpdnet_psma_fs50_f${FOLD_ID}_${STAMP}.txt"
