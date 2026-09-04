#!/usr/bin/env bash
# ICLR2026 baseline1：仅 FDG train（分层划分）· Dataset228 2ch · 3d_fullres
# GPU0+1+3 · 全局 bs=6（每卡2）· tr=70 · val=10 · 3000 ep · best=val_loss（val 仅 loss，无 Pseudo/EMA）
#
#   export TASK1_BASE=/media/ybwang/data1/PSMA-DATA
#   bash ICLR2026/run/run_baseline1_fdg_2ch_fullres_3000ep_bg.sh
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
TOTAL_EPOCHS="${TASK1_NUM_EPOCHS:-3000}"
# 强制 fold 为 0..4 或 all；拒绝环境里残留的路径污染
_raw_fold="${FOLD:-0}"
if [[ "${_raw_fold}" =~ ^(all|[0-4])$ ]]; then
  FOLD="${_raw_fold}"
else
  echo "[baseline1] WARN ignore invalid FOLD=${_raw_fold}；改用 0" >&2
  FOLD=0
fi
TRAIN_ITERS="${TASK1_TRAIN_ITERS_PER_EPOCH:-70}"
VAL_ITERS="${TASK1_VAL_ITERS_PER_EPOCH:-10}"

SPLITS_JSON="${TASK1_SPLITS_FINAL_JSON:-${ROOT}/ICLR2026/data/splits_baseline1_fdg_nnunet.json}"
[[ -f "${SPLITS_JSON}" ]] || {
  echo "[error] missing splits: ${SPLITS_JSON}" >&2
  echo "  run: python3 ICLR2026/scripts/export_baseline1_fdg_splits.py" >&2
  exit 1
}

PREP_MARKER="${WORK}/nnUNet_preprocessed/${DS}/nnUNetPlans_${CONFIG}"
[[ -d "${PREP_MARKER}" ]] || {
  echo "[error] 缺少预处理 ${PREP_MARKER}" >&2
  exit 1
}

# Keep STAMP for baseline1 / nnUNet MIM / competition scratch FDG.
_req_stamp="${TASK1_NNUNET_RESULTS_STAMP_NAME:-}"
_comp_scratch_methods="hemingduo_scratch|chenyixin_scratch|hemingduo|chenyixin"
if [[ -n "${_req_stamp}" && ( "${_req_stamp}" == *iclr2026_baseline1* || "${_req_stamp}" == *iclr2026_nnunet_mim* || "${_req_stamp}" == *iclr2026_hemingduo* || "${_req_stamp}" == *iclr2026_chenyixin* ) ]]; then
  NEW_STAMP="${_req_stamp}"
else
  if [[ -n "${_req_stamp}" ]]; then
    echo "[baseline1] ignore non-baseline1 STAMP=${_req_stamp}；生成新 STAMP" >&2
  fi
  if [[ "${TASK1_BOARD_METHOD:-}" == "nnunet_mim" ]]; then
    NEW_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_nnunet_mim_fdg_2ch_fullres_gpu013_bs6_tr${TRAIN_ITERS}_val${VAL_ITERS}_${TOTAL_EPOCHS}ep"
  elif [[ "${TASK1_BOARD_METHOD:-}" =~ ^(${_comp_scratch_methods})$ ]]; then
    NEW_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_${TASK1_BOARD_METHOD}_fdg_2ch_fullres_gpu013_bs6_tr${TRAIN_ITERS}_val${VAL_ITERS}_${TOTAL_EPOCHS}ep"
  else
    NEW_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr${TRAIN_ITERS}_val${VAL_ITERS}_${TOTAL_EPOCHS}ep"
  fi
fi
TF="${TRAINER}__${PLANS_ID}__${CONFIG}"
RESTART="ICLR2026/run/run_baseline1_fdg_2ch_fullres_3000ep_bg.sh"

# 后 1/3：从 ep2000 起额外 PSMA val（监控，不参与 FDG best）；step 同 FDG val=10
PSMA_VAL_JSON="${TASK1_PSMA_VAL_CASES_JSON:-${ROOT}/ICLR2026/data/splits_baseline1_psma_val.json}"
[[ -f "${PSMA_VAL_JSON}" ]] || {
  echo "[error] missing PSMA val cases: ${PSMA_VAL_JSON}" >&2
  echo "  run: python3 ICLR2026/scripts/export_baseline1_psma_val.py" >&2
  exit 1
}
export TASK1_PSMA_VAL_ENABLE="${TASK1_PSMA_VAL_ENABLE:-1}"
export TASK1_PSMA_VAL_FROM_EPOCH="${TASK1_PSMA_VAL_FROM_EPOCH:-2000}"
# 后 1/3：FDG + PSMA val step 均改为 50（前半仍用 VAL_ITERS=10）
export TASK1_VAL_ITERS_LATE_FROM_EPOCH="${TASK1_VAL_ITERS_LATE_FROM_EPOCH:-${TASK1_PSMA_VAL_FROM_EPOCH}}"
export TASK1_VAL_ITERS_LATE_PER_EPOCH="${TASK1_VAL_ITERS_LATE_PER_EPOCH:-50}"
export TASK1_PSMA_VAL_ITERS_PER_EPOCH="${TASK1_PSMA_VAL_ITERS_PER_EPOCH:-${TASK1_VAL_ITERS_LATE_PER_EPOCH}}"
export TASK1_PSMA_VAL_CASES_JSON="${PSMA_VAL_JSON}"

echo "[baseline1] STAMP=${NEW_STAMP}"
echo "[baseline1] splits=${SPLITS_JSON} fold=${FOLD} best_by=val_loss tr=${TRAIN_ITERS} val=${VAL_ITERS}"
echo "[baseline1] late_val from_ep=${TASK1_VAL_ITERS_LATE_FROM_EPOCH} fdg_iters=${TASK1_VAL_ITERS_LATE_PER_EPOCH} psma_iters=${TASK1_PSMA_VAL_ITERS_PER_EPOCH} cases=${PSMA_VAL_JSON}"

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
export TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH="${TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH:-99}"
export TASK1_CONTINUE_TRAINING="${TASK1_CONTINUE_TRAINING:-0}"
export TASK1_CONTINUE_PICK_NEWER="${TASK1_CONTINUE_PICK_NEWER:-0}"
# Fresh run: do not default continue-from-latest (GUARD_EXTRA_ENV still forces CONTINUE=1 on resume)
export TASK1_CONTINUE_FROM_LATEST="${TASK1_CONTINUE_FROM_LATEST:-0}"
export TASK1_CONTINUE_FROM_BEST="${TASK1_CONTINUE_FROM_BEST:-0}"
export TASK1_RAM_SHARD_ENABLE=0
export TASK1_SPLITS_FINAL_JSON="${SPLITS_JSON}"
export TASK1_BEST_BY="${TASK1_BEST_BY:-val_loss}"
# best=val_loss：跳过 hard Pseudo dice / EMA；图也不画 val ema
export TASK1_VAL_LOSS_ONLY="${TASK1_VAL_LOSS_ONLY:-1}"
export TASK1_LOSS_PLOT_VAL_EMA="${TASK1_LOSS_PLOT_VAL_EMA:-0}"
export TRAINER
export CONFIG
export PLANS_ID
export FOLD
export DATASET_ID
export TASK1_DATASET_NAME="${DS}"
export TASK1_NNUNET_RESULTS_STAMP_SUBDIR=1
export TASK1_NNUNET_RESULTS_STAMP_NAME="${NEW_STAMP}"
# loss / 监控图落在 ICLR2026/vis（不写竞赛 01_train_vis）
ICLR_VIS="${TASK1_LOSS_OUT_DIR:-${ROOT}/ICLR2026/vis}"
mkdir -p "${ICLR_VIS}"
export TASK1_LOSS_OUT_DIR="${ICLR_VIS}"
export TASK1_LOSS_OUT_DIR_EXTRA="${TASK1_LOSS_OUT_DIR_EXTRA:-none}"
export TASK1_LOSS_OUT_NAME="${TASK1_LOSS_OUT_NAME:-loss_curve_iclr2026_baseline1_fdg_${NEW_STAMP}.png}"
export TASK1_LIVE_LOSS_PLOT="${TASK1_LIVE_LOSS_PLOT:-1}"
export TASK1_LOSS_MERGE_ALL_LOGS=1
export TASK1_LOSS_MERGE_LAST_K=0
export TASK1_LOSS_MERGE_MAX_FILE_BYTES=0
# 横轴随已训 epoch 自动伸缩（X_FOLLOW 覆盖容器内可能残留的 X_MAX=3000）
export TASK1_LOSS_PLOT_X_FOLLOW=1
export TASK1_LOSS_PLOT_X_MAX_EPOCHS=""
# 画图：该 epoch 之前只画 train（蓝线），不画 FDG/PSMA val
export TASK1_LOSS_PLOT_VAL_FROM_EPOCH="${TASK1_LOSS_PLOT_VAL_FROM_EPOCH:-2275}"
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

# GUARD resume always CONTINUE=1; fresh launch uses exports above (0)
EXTRA_ENV="TASK1_NUM_EPOCHS=${TOTAL_EPOCHS},TASK1_TRAIN_ITERS_PER_EPOCH=${TASK1_TRAIN_ITERS_PER_EPOCH},TASK1_VAL_ITERS_PER_EPOCH=${TASK1_VAL_ITERS_PER_EPOCH},TASK1_VAL_EVERY_N_EPOCHS=${TASK1_VAL_EVERY_N_EPOCHS:-1},TASK1_VAL_ITERS_LATE_FROM_EPOCH=${TASK1_VAL_ITERS_LATE_FROM_EPOCH},TASK1_VAL_ITERS_LATE_PER_EPOCH=${TASK1_VAL_ITERS_LATE_PER_EPOCH},TASK1_FIXED_BATCH_3D_FULLRES=${TASK1_FIXED_BATCH_3D_FULLRES},TASK1_N_PROC_DA=${TASK1_N_PROC_DA},TASK1_SEGMENT_CHECKPOINT=0,TASK1_PERIODIC_CHECKPOINT_EVERY=0,TASK1_ALWAYS_SAVE_LATEST=1,TASK1_SAVE_LATEST_EVERY=${TASK1_SAVE_LATEST_EVERY},TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH=${TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH},TASK1_RAM_SHARD_ENABLE=0,TASK1_SPLITS_FINAL_JSON=${SPLITS_JSON},TASK1_BEST_BY=${TASK1_BEST_BY},TASK1_VAL_LOSS_ONLY=${TASK1_VAL_LOSS_ONLY},TASK1_PSMA_VAL_ENABLE=${TASK1_PSMA_VAL_ENABLE},TASK1_PSMA_VAL_FROM_EPOCH=${TASK1_PSMA_VAL_FROM_EPOCH},TASK1_PSMA_VAL_ITERS_PER_EPOCH=${TASK1_PSMA_VAL_ITERS_PER_EPOCH},TASK1_PSMA_VAL_CASES_JSON=${TASK1_PSMA_VAL_CASES_JSON},TASK1_LOSS_PLOT_VAL_EMA=${TASK1_LOSS_PLOT_VAL_EMA},TASK1_LOSS_PLOT_VAL_FROM_EPOCH=${TASK1_LOSS_PLOT_VAL_FROM_EPOCH},TASK1_LOSS_PLOT_X_FOLLOW=1,TASK1_LOSS_PLOT_X_MAX_EPOCHS=,TASK1_LOSS_OUT_DIR=${TASK1_LOSS_OUT_DIR},TASK1_LOSS_OUT_DIR_EXTRA=${TASK1_LOSS_OUT_DIR_EXTRA},TASK1_LOSS_OUT_NAME=${TASK1_LOSS_OUT_NAME},TASK1_CONTINUE_TRAINING=1,TASK1_CONTINUE_FROM_LATEST=1,TASK1_CONTINUE_PICK_NEWER=1,DATASET_ID=${DATASET_ID},TASK1_DATASET_NAME=${DS},TASK1_NNUNET_RESULTS_STAMP_NAME=${NEW_STAMP},TASK1_NNUNET_RESULTS_STAMP_SUBDIR=1,CONFIG=${CONFIG},FOLD=${FOLD},TRAINER=${TRAINER},PLANS_ID=${PLANS_ID},TASK1_SKIP_RAW_PREP=1,TASK1_SKIP_PLAN_PREP=1,TASK1_STOP_AFTER_PREP=0,TASK1_TRAIN_NUM_GPUS=${TASK1_TRAIN_NUM_GPUS},TASK1_DOCKER_GPUS=${TASK1_DOCKER_GPUS},TASK1_CUDA_VISIBLE_DEVICES=${TASK1_CUDA_VISIBLE_DEVICES},TASK1_BOARD_METHOD=${TASK1_BOARD_METHOD:-}"

TASK1_GUARD_STAMP="${NEW_STAMP}" \
TASK1_NNUNET_RESULTS_STAMP_NAME="${NEW_STAMP}" \
TASK1_GUARD_TRAINER_FOLDER="${TF}" \
TASK1_GUARD_DATASET_DIR="${DS}" \
TASK1_GUARD_TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
TASK1_GUARD_RESTART_SCRIPT="${RESTART}" \
TASK1_GUARD_REQUIRE_ARM=1 \
TASK1_GUARD_EXTRA_ENV="${EXTRA_ENV}" \
FOLD="${FOLD}" \
  bash "${ROOT}/run_task/run_task1_train_auto_resume_guard_bg.sh" || true

TASK1_CRASH_MONITOR_STAGE=baseline1_train_start \
TASK1_NNUNET_RESULTS_STAMP_NAME="${NEW_STAMP}" \
  bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true

export TASK1_PREFLIGHT_STEP="iclr2026-baseline1-fdg-2ch-fullres"
bash "${ROOT}/scripts/task1_gpu_train_preflight.sh" || true

bash "${ROOT}/other/task1_train_nnunet_from_dataset1.sh"

sleep 25
TASK1_CRASH_MONITOR_STAGE=baseline1_train_running \
TASK1_NNUNET_RESULTS_STAMP_NAME="${NEW_STAMP}" \
TASK1_CRASH_MONITOR_ARM_SEC="${TASK1_CRASH_MONITOR_ARM_SEC:-86400}" \
  bash "${ROOT}/scripts/task1_crash_monitor_arm.sh" || true

FOLD_DIR="${WORK}/nnUNet_results/${NEW_STAMP}/${DS}/${TF}/fold_${FOLD}"
MANIFEST="${ICLR_VIS}/iclr2026_baseline1_fdg_${NEW_STAMP}.txt"
{
  echo "job=iclr2026_baseline1_fdg_2ch_fullres"
  echo "STAMP=${NEW_STAMP}"
  echo "DATASET_ID=${DATASET_ID}"
  echo "DATASET_NAME=${DS}"
  echo "TRAINER=${TRAINER}"
  echo "CONFIG=${CONFIG}"
  echo "backbone=PlainConvUNet"
  echo "channels=CT+PET"
  echo "data=FDG_train_only_from_stratified_70_10_20"
  echo "splits=${SPLITS_JSON}"
  echo "fold=${FOLD}"
  echo "best_by=${TASK1_BEST_BY}"
  echo "val_loss_only=${TASK1_VAL_LOSS_ONLY}"
  echo "psma_val_enable=${TASK1_PSMA_VAL_ENABLE}"
  echo "psma_val_from_epoch=${TASK1_PSMA_VAL_FROM_EPOCH}"
  echo "psma_val_iters=${TASK1_PSMA_VAL_ITERS_PER_EPOCH}"
  echo "psma_val_cases=${TASK1_PSMA_VAL_CASES_JSON}"
  echo "val_iters_late_from_epoch=${TASK1_VAL_ITERS_LATE_FROM_EPOCH}"
  echo "val_iters_late_per_epoch=${TASK1_VAL_ITERS_LATE_PER_EPOCH}"
  echo "loss_plot_val_ema=${TASK1_LOSS_PLOT_VAL_EMA}"
  echo "gpus=3 physical=0,1,3 global_batch=${TASK1_FIXED_BATCH_3D_FULLRES}"
  echo "train_iters=${TASK1_TRAIN_ITERS_PER_EPOCH} val_iters=${TASK1_VAL_ITERS_PER_EPOCH} (late=${TASK1_VAL_ITERS_LATE_PER_EPOCH}@${TASK1_VAL_ITERS_LATE_FROM_EPOCH})"
  echo "num_epochs=${TOTAL_EPOCHS}"
  echo "fold_dir=${FOLD_DIR}"
  echo "loss_out_dir=${TASK1_LOSS_OUT_DIR}"
  echo "loss_png=${TASK1_LOSS_OUT_DIR}/${TASK1_LOSS_OUT_NAME}"
} | tee "${MANIFEST}"

if [[ "${TASK1_BOARD_METHOD:-}" == "nnunet_mim" ]]; then
  echo "${NEW_STAMP}" > "${ICLR_VIS}/nnunet_mim_fdg_LAST_STAMP.txt"
elif [[ "${TASK1_BOARD_METHOD:-}" =~ ^(hemingduo_scratch|chenyixin_scratch|hemingduo|chenyixin)$ ]]; then
  echo "${NEW_STAMP}" > "${ICLR_VIS}/${TASK1_BOARD_METHOD}_fdg_LAST_STAMP.txt"
else
  echo "${NEW_STAMP}" > "${ICLR_VIS}/baseline1_fdg_LAST_STAMP.txt"
fi
echo "[baseline1] launched manifest=${MANIFEST}"
echo "  loss_png=${TASK1_LOSS_OUT_DIR}/${TASK1_LOSS_OUT_NAME}"
echo "  stop: TASK1_NNUNET_RESULTS_STAMP_NAME=${NEW_STAMP} bash scripts/task1_stop_train_and_resume.sh"
