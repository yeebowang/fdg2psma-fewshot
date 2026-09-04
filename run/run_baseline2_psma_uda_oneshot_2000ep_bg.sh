#!/usr/bin/env bash
# ICLR2026 baseline2 oneshot：FDG best 初始化 → 一次伪标 → PSMA 伪标微调 2000ep
# 无 round 循环。train 70 / val 50 · GPU 0,1,3 · bs=6 · best=val_loss · LR=1e-3
#
#   export TASK1_BASE=/media/ybwang/data1/PSMA-DATA
#   bash ICLR2026/run/run_baseline2_psma_uda_oneshot_2000ep_bg.sh
#
# 可选复用已有伪标 b2nd：
#   export TASK1_PSEUDO_SEG_B2ND_DIR=.../round_000/pseudo_seg_b2nd
#   export TASK1_UDA_SKIP_PSEUDO=1
#
# 停止：TASK1_NNUNET_RESULTS_STAMP_NAME=<STAMP> bash scripts/task1_stop_train_and_resume.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK1_REPO_ROOT="${ROOT}"
export TASK1_BASE="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
export WORK_DIR="${WORK_DIR:-${TASK1_BASE}/task1_train_workspace}"
WORK="${WORK_DIR}"
VIS="${WORK}/01_train_vis"
mkdir -p "${VIS}/log"

DATASET_ID="${DATASET_ID:-228}"
DS="Dataset${DATASET_ID}_AutoPETIV_Task1_2ch"
TRAINER="${TRAINER:-nnUNetTrainer_Task1StdTrainVal50}"
PLANS_ID="${PLANS_ID:-nnUNetPlans}"
CONFIG="${CONFIG:-3d_fullres}"
TOTAL_EPOCHS="${TASK1_NUM_EPOCHS:-2000}"
_raw_fold="${FOLD:-0}"
if [[ "${_raw_fold}" =~ ^(all|[0-4])$ ]]; then
  FOLD="${_raw_fold}"
else
  echo "[baseline2-oneshot] WARN ignore invalid FOLD=${_raw_fold}；改用 0" >&2
  FOLD=0
fi
TF="${TRAINER}__${PLANS_ID}__${CONFIG}"
TRAIN_ITERS="${TASK1_TRAIN_ITERS_PER_EPOCH:-70}"
VAL_ITERS="${TASK1_VAL_ITERS_PER_EPOCH:-50}"

SPLITS_JSON="${TASK1_SPLITS_FINAL_JSON:-${ROOT}/ICLR2026/data/splits_baseline2_psma_uda_nnunet.json}"
[[ -f "${SPLITS_JSON}" ]] || {
  echo "[error] missing splits: ${SPLITS_JSON}" >&2
  exit 1
}
PREP_DIR="${WORK}/nnUNet_preprocessed/${DS}/nnUNetPlans_${CONFIG}"
[[ -d "${PREP_DIR}" ]] || { echo "[error] missing prep ${PREP_DIR}" >&2; exit 1; }

FDG_STAMP="${TASK1_UDA_FDG_STAMP:-20260810_104431_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val10_3000ep}"
FDG_BEST="${TASK1_UDA_FDG_BEST:-${WORK}/nnUNet_results/${FDG_STAMP}/${DS}/${TF}/fold_${FOLD}/checkpoint_best.pth}"
[[ -f "${FDG_BEST}" ]] || { echo "[error] missing FDG best: ${FDG_BEST}" >&2; exit 1; }

_req="${TASK1_NNUNET_RESULTS_STAMP_NAME:-}"
if [[ -n "${_req}" && "${_req}" == *iclr2026_baseline2*oneshot* ]]; then
  NEW_STAMP="${_req}"
elif [[ -n "${_req}" && "${_req}" == *iclr2026_baseline2_psma_uda_oneshot* ]]; then
  NEW_STAMP="${_req}"
else
  if [[ -n "${_req}" ]]; then
    echo "[baseline2-oneshot] ignore STAMP=${_req}；生成新 STAMP" >&2
  fi
  NEW_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_baseline2_psma_uda_oneshot_2ch_gpu013_bs6_tr${TRAIN_ITERS}_val${VAL_ITERS}_${TOTAL_EPOCHS}ep"
fi

ICLR_VIS="${TASK1_LOSS_OUT_DIR:-${ROOT}/ICLR2026/vis}"
mkdir -p "${ICLR_VIS}"
UDA_ROOT="${WORK}/nnUNet_results/${NEW_STAMP}"
mkdir -p "${UDA_ROOT}"
PSEUDO_ROOT="${UDA_ROOT}/pseudo"
PRED_OUT="${PSEUDO_ROOT}/pred"
LABELS_DIR="${PSEUDO_ROOT}/labelsTr_pseudo"
SEG_B2ND="${TASK1_PSEUDO_SEG_B2ND_DIR:-${PSEUDO_ROOT}/pseudo_seg_b2nd}"
PRIOR_JSON="${UDA_ROOT}/psma_uda_prior_state.json"
IMAGE_TAG="${IMAGE_TAG:-autopet_baseline}"
RESTART="ICLR2026/run/run_baseline2_psma_uda_oneshot_2000ep_bg.sh"

export TASK1_STOP_AFTER_PREP=0
export TASK1_SKIP_RAW_PREP=1
export TASK1_SKIP_PLAN_PREP=1
export TASK1_FORCE_PLAN_PREP=0
export TASK1_NUM_EPOCHS="${TOTAL_EPOCHS}"
export TASK1_TRAIN_ITERS_PER_EPOCH="${TRAIN_ITERS}"
export TASK1_VAL_ITERS_PER_EPOCH="${VAL_ITERS}"
export TASK1_FIXED_BATCH_3D_FULLRES="${TASK1_FIXED_BATCH_3D_FULLRES:-6}"
export TASK1_N_PROC_DA="${TASK1_N_PROC_DA:-6}"
export TASK1_SEGMENT_CHECKPOINT=0
export TASK1_PERIODIC_CHECKPOINT_EVERY=0
export TASK1_ALWAYS_SAVE_LATEST=1
export TASK1_SAVE_LATEST_EVERY="${TASK1_SAVE_LATEST_EVERY:-25}"
export TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH="${TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH:-0}"
# 强制新训（忽略 shell 残留的 CONTINUE=1）
export TASK1_CONTINUE_TRAINING=0
export TASK1_CONTINUE_FROM_LATEST=0
export TASK1_CONTINUE_FROM_BEST=0
export TASK1_CONTINUE_PICK_NEWER=0
export TASK1_RAM_SHARD_ENABLE=0
export TASK1_SPLITS_FINAL_JSON="${SPLITS_JSON}"
export TASK1_BEST_BY="${TASK1_BEST_BY:-val_loss}"
export TASK1_VAL_LOSS_ONLY="${TASK1_VAL_LOSS_ONLY:-1}"
export TASK1_LOSS_PLOT_VAL_EMA=0
# 清掉 FDG 等残留的 VAL_FROM（否则会把 oneshot 的 val 全滤掉）
export TASK1_LOSS_PLOT_VAL_FROM_EPOCH=
export TASK1_PSMA_VAL_ENABLE=0
export TASK1_INITIAL_LR="${TASK1_INITIAL_LR:-0.001}"
export TRAINER CONFIG PLANS_ID FOLD DATASET_ID
export TASK1_DATASET_NAME="${DS}"
export TASK1_NNUNET_RESULTS_STAMP_SUBDIR=1
export TASK1_NNUNET_RESULTS_STAMP_NAME="${NEW_STAMP}"
export TASK1_TRAINER_PY="${TASK1_TRAINER_PY:-${ROOT}/nnunet_ext_trainers/nnUNetTrainer_Task1StdTrainVal50.py}"
export TASK1_TRAIN_NUM_GPUS="${TASK1_TRAIN_NUM_GPUS:-3}"
export TASK1_DOCKER_GPUS="${TASK1_DOCKER_GPUS:-device=0,1,3}"
export TASK1_CUDA_VISIBLE_DEVICES="${TASK1_CUDA_VISIBLE_DEVICES:-0,1,3}"
export TASK1_PREFLIGHT_GPUS="${TASK1_PREFLIGHT_GPUS:-0 1 3}"
_nproc_da="${TASK1_N_PROC_DA}"
_shm_g=$(( _nproc_da * 3 ))
(( _shm_g < 16 )) && _shm_g=16
export TASK1_DOCKER_SHM="${TASK1_DOCKER_SHM:-${_shm_g}g}"
export TASK1_DOCKER_BACKGROUND="${TASK1_DOCKER_BACKGROUND:-1}"
export TASK1_LOSS_OUT_DIR="${ICLR_VIS}"
export TASK1_LOSS_OUT_DIR_EXTRA=none
export TASK1_LOSS_OUT_NAME="${TASK1_LOSS_OUT_NAME:-loss_curve_iclr2026_baseline2_oneshot_${NEW_STAMP}.png}"
export TASK1_LIVE_LOSS_PLOT=1
export TASK1_LOSS_MERGE_ALL_LOGS=1
export TASK1_LOSS_PLOT_X_FOLLOW=1
export TASK1_LOSS_PLOT_SHOW_ETA=1
export TASK1_LOSS_PLOT_SEED_EMPTY=1
export TASK1_LOSS_PLOT_X_MAX_EPOCHS="${TOTAL_EPOCHS}"
export TASK1_LR_SCHEDULE_NUM_EPOCHS="${TOTAL_EPOCHS}"
export TASK1_PRETRAINED_WEIGHTS="${FDG_BEST}"

echo "[baseline2-oneshot] STAMP=${NEW_STAMP}"
echo "[baseline2-oneshot] FDG_BEST=${FDG_BEST}"
echo "[baseline2-oneshot] splits=${SPLITS_JSON} ep=${TOTAL_EPOCHS} tr=${TRAIN_ITERS} val=${VAL_ITERS} lr=${TASK1_INITIAL_LR}"
echo "[baseline2-oneshot] pseudo_b2nd=${SEG_B2ND}"
echo "[baseline2-oneshot] loss=${ICLR_VIS}/${TASK1_LOSS_OUT_NAME}"

# ---- 伪标（可跳过）----
SKIP_PSEUDO="${TASK1_UDA_SKIP_PSEUDO:-0}"
n_b2nd=0
if [[ -d "${SEG_B2ND}" ]]; then
  n_b2nd="$(find "${SEG_B2ND}" -maxdepth 1 -name '*_seg.b2nd' 2>/dev/null | wc -l | tr -d ' ')"
fi
if [[ "${SKIP_PSEUDO}" == "1" ]] || [[ "${n_b2nd}" -ge 400 ]]; then
  echo "[baseline2-oneshot] reuse pseudo b2nd (${n_b2nd} seg) → ${SEG_B2ND}"
  SKIP_PSEUDO=1
else
  SKIP_PSEUDO=0
fi

if [[ "${SKIP_PSEUDO}" != "1" ]]; then
  mkdir -p "${PRED_OUT}/pred" "${LABELS_DIR}" "${SEG_B2ND}"
  chmod -R a+rwX "${PSEUDO_ROOT}" "${UDA_ROOT}" 2>/dev/null || true
  export TASK1_UDA_CKPT="${FDG_BEST}"
  export TASK1_UDA_PRED_OUT="${PRED_OUT}"
  export TASK1_UDA_CASES_JSON="${SPLITS_JSON}"
  export TASK1_UDA_NNUNET_RESULTS="$(dirname "$(dirname "$(dirname "$(dirname "${FDG_BEST}")")")")"
  echo "[baseline2-oneshot] predict PSMA train …"
  bash "${ROOT}/ICLR2026/scripts/psma_uda_predict_train.sh"
  echo "[baseline2-oneshot] make pseudo …"
  docker run --rm \
    -v "${ROOT}:${ROOT}" -v "${WORK}:${WORK}" -v "${TASK1_BASE}:${TASK1_BASE}" \
    -w "${ROOT}" --entrypoint python3 "${IMAGE_TAG}" \
    ICLR2026/scripts/psma_uda_make_pseudo_labels.py \
      --pred-dir "${PRED_OUT}/pred" \
      --cases-json "${SPLITS_JSON}" \
      --out-labels-dir "${LABELS_DIR}" \
      --prior-state "${PRIOR_JSON}" \
      --round 0 \
      --rounds-total 1 \
      --lambda-start "${TASK1_UDA_LAMBDA_START:-0.1}" \
      --lambda-end "${TASK1_UDA_LAMBDA_END:-0.1}"
  echo "[baseline2-oneshot] nii→b2nd …"
  docker run --rm \
    -v "${ROOT}:${ROOT}" -v "${WORK}:${WORK}" -v "${TASK1_BASE}:${TASK1_BASE}" \
    -w "${ROOT}" --entrypoint python3 "${IMAGE_TAG}" \
    ICLR2026/scripts/psma_uda_pseudo_nii_to_seg_b2nd.py \
      --prep-dir "${PREP_DIR}" \
      --labels-dir "${LABELS_DIR}" \
      --out-seg-b2nd-dir "${SEG_B2ND}" \
      --cases-json "${SPLITS_JSON}"
fi

export TASK1_PSEUDO_SEG_B2ND_DIR="${SEG_B2ND}"
[[ -d "${TASK1_PSEUDO_SEG_B2ND_DIR}" ]] || {
  echo "[error] missing TASK1_PSEUDO_SEG_B2ND_DIR=${TASK1_PSEUDO_SEG_B2ND_DIR}" >&2
  exit 1
}

EXTRA_ENV="TASK1_NUM_EPOCHS=${TOTAL_EPOCHS},TASK1_TRAIN_ITERS_PER_EPOCH=${TRAIN_ITERS},TASK1_VAL_ITERS_PER_EPOCH=${VAL_ITERS},TASK1_FIXED_BATCH_3D_FULLRES=${TASK1_FIXED_BATCH_3D_FULLRES},TASK1_N_PROC_DA=${TASK1_N_PROC_DA},TASK1_SEGMENT_CHECKPOINT=0,TASK1_PERIODIC_CHECKPOINT_EVERY=0,TASK1_ALWAYS_SAVE_LATEST=1,TASK1_SAVE_LATEST_EVERY=${TASK1_SAVE_LATEST_EVERY},TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH=${TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH},TASK1_RAM_SHARD_ENABLE=0,TASK1_SPLITS_FINAL_JSON=${SPLITS_JSON},TASK1_BEST_BY=${TASK1_BEST_BY},TASK1_VAL_LOSS_ONLY=${TASK1_VAL_LOSS_ONLY},TASK1_PSMA_VAL_ENABLE=0,TASK1_LOSS_PLOT_VAL_EMA=0,TASK1_LOSS_PLOT_VAL_FROM_EPOCH=,TASK1_LOSS_OUT_DIR=${ICLR_VIS},TASK1_LOSS_OUT_DIR_EXTRA=none,TASK1_LOSS_OUT_NAME=${TASK1_LOSS_OUT_NAME},TASK1_LIVE_LOSS_PLOT=1,TASK1_LOSS_MERGE_ALL_LOGS=1,TASK1_LOSS_PLOT_X_FOLLOW=1,TASK1_LOSS_PLOT_SHOW_ETA=1,TASK1_LOSS_PLOT_SEED_EMPTY=1,TASK1_LOSS_PLOT_X_MAX_EPOCHS=${TOTAL_EPOCHS},TASK1_LR_SCHEDULE_NUM_EPOCHS=${TOTAL_EPOCHS},TASK1_INITIAL_LR=${TASK1_INITIAL_LR},TASK1_PRETRAINED_WEIGHTS=${FDG_BEST},TASK1_PSEUDO_SEG_B2ND_DIR=${SEG_B2ND},TASK1_NNUNET_RESULTS_STAMP_NAME=${NEW_STAMP},TASK1_NNUNET_RESULTS_STAMP_SUBDIR=1,DATASET_ID=${DATASET_ID},TASK1_DATASET_NAME=${DS},CONFIG=${CONFIG},FOLD=${FOLD},TRAINER=${TRAINER},PLANS_ID=${PLANS_ID},TASK1_SKIP_RAW_PREP=1,TASK1_SKIP_PLAN_PREP=1,TASK1_STOP_AFTER_PREP=0,TASK1_TRAIN_NUM_GPUS=${TASK1_TRAIN_NUM_GPUS},TASK1_DOCKER_GPUS=${TASK1_DOCKER_GPUS},TASK1_CUDA_VISIBLE_DEVICES=${TASK1_CUDA_VISIBLE_DEVICES},TASK1_DOCKER_BACKGROUND=1,TASK1_CONTINUE_TRAINING=0,TASK1_CONTINUE_FROM_LATEST=0,TASK1_CONTINUE_FROM_BEST=0,TASK1_CONTINUE_PICK_NEWER=0"

TASK1_GUARD_STAMP="${NEW_STAMP}" \
TASK1_GUARD_CONFIG="${CONFIG}" \
TASK1_GUARD_TRAINER_FOLDER="${TF}" \
TASK1_GUARD_DATASET_DIR="${DS}" \
TASK1_GUARD_TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
TASK1_GUARD_RESTART_SCRIPT="${RESTART}" \
TASK1_GUARD_REQUIRE_ARM=1 \
TASK1_GUARD_EXTRA_ENV="${EXTRA_ENV}" \
  bash "${ROOT}/run_task/run_task1_train_auto_resume_guard_bg.sh" || true

TASK1_CRASH_MONITOR_STAGE="baseline2_oneshot_before_train" \
TASK1_NNUNET_RESULTS_STAMP_NAME="${NEW_STAMP}" \
  bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true

export TASK1_PREFLIGHT_STEP="iclr2026-baseline2-oneshot"
bash "${ROOT}/scripts/task1_gpu_train_preflight.sh" || true
sleep 2

TASK1_CRASH_MONITOR_STAGE="baseline2_oneshot_train_running" \
TASK1_NNUNET_RESULTS_STAMP_NAME="${NEW_STAMP}" \
TASK1_CRASH_MONITOR_ARM_SEC="${TASK1_CRASH_MONITOR_ARM_SEC:-86400}" \
  bash "${ROOT}/scripts/task1_crash_monitor_arm.sh" || true

export TASK1_PREFLIGHT_STEP="iclr2026-baseline2-oneshot"
bash "${ROOT}/other/task1_train_nnunet_from_dataset1.sh"

MANIFEST="${ICLR_VIS}/iclr2026_baseline2_oneshot_${NEW_STAMP}.txt"
{
  echo "job=iclr2026_baseline2_psma_uda_oneshot"
  echo "STAMP=${NEW_STAMP}"
  echo "FDG_BEST=${FDG_BEST}"
  echo "epochs=${TOTAL_EPOCHS} train_iters=${TRAIN_ITERS} val_iters=${VAL_ITERS}"
  echo "initial_lr=${TASK1_INITIAL_LR}"
  echo "splits=${SPLITS_JSON}"
  echo "pseudo_b2nd=${SEG_B2ND}"
  echo "loss_png=${ICLR_VIS}/${TASK1_LOSS_OUT_NAME}"
} | tee "${MANIFEST}"

echo "[baseline2-oneshot] launched train STAMP=${NEW_STAMP}"
echo "  loss_png=${ICLR_VIS}/${TASK1_LOSS_OUT_NAME}"
echo "  stop: TASK1_NNUNET_RESULTS_STAMP_NAME=${NEW_STAMP} bash scripts/task1_stop_train_and_resume.sh"
