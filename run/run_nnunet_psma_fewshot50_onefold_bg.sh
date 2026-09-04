#!/usr/bin/env bash
# Single-fold nnUNet PSMA fewshot50 finetune (1 GPU, tr=70, val=70 every 20ep).
# Required env: FOLD_ID (2/5/8), GPU_ID, PARENT_STAMP (or TASK1_NNUNET_RESULTS_STAMP_NAME)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK1_REPO_ROOT="${ROOT}"
export TASK1_BASE="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
export WORK_DIR="${WORK_DIR:-${TASK1_BASE}/task1_train_workspace}"
WORK="${WORK_DIR}"
ICLR_VIS="${TASK1_LOSS_OUT_DIR:-${ROOT}/ICLR2026/vis}"
SPLIT_DIR="${TASK1_FEWSHOT_SPLIT_DIR:-${ROOT}/ICLR2026/data/splits_mae_psma_fewshot50_9fold}"
mkdir -p "${ICLR_VIS}"

FOLD_ID="${FOLD_ID:?need FOLD_ID}"
GPU_ID="${GPU_ID:?need GPU_ID}"
FEWSHOT_N="${TASK1_FEWSHOT_N:-50}"
DATASET_ID="${DATASET_ID:-228}"
DS="Dataset${DATASET_ID}_AutoPETIV_Task1_2ch"
TRAINER="${TRAINER:-nnUNetTrainer_Task1StdTrainVal50}"
PLANS_ID="${PLANS_ID:-nnUNetPlans}"
CONFIG="${CONFIG:-3d_fullres}"
TF="${TRAINER}__${PLANS_ID}__${CONFIG}"
TOTAL_EPOCHS="${TASK1_NUM_EPOCHS:-300}"
TRAIN_ITERS="${TASK1_TRAIN_ITERS_PER_EPOCH:-70}"
VAL_ITERS="${TASK1_FS50_VAL_ITERS:-${TASK1_VAL_ITERS_PER_EPOCH:-70}}"
VAL_EVERY="${TASK1_FS50_VAL_EVERY_N_EPOCHS:-${TASK1_VAL_EVERY_N_EPOCHS:-20}}"
BATCH="${TASK1_FIXED_BATCH_3D_FULLRES:-6}"
# PSMA fewshot：按 val_loss 选 best（不看 ema dice）；默认 val_loss_only=1
BEST_BY="${TASK1_BEST_BY:-val_loss}"
VAL_LOSS_ONLY="${TASK1_VAL_LOSS_ONLY:-1}"

FDG_STAMP="${TASK1_UDA_FDG_STAMP:-20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep}"
FDG_BEST="${TASK1_UDA_FDG_BEST:-}"
if [[ -z "${FDG_BEST}" ]]; then
  _fold="${WORK}/nnUNet_results/${FDG_STAMP}/${DS}/${TF}/fold_0"
  for _c in checkpoint_final.pth checkpoint_latest.pth checkpoint_best.pth; do
    [[ -f "${_fold}/${_c}" ]] && { FDG_BEST="${_fold}/${_c}"; break; }
  done
fi
[[ -n "${FDG_BEST}" && -f "${FDG_BEST}" ]] || { echo "[error] missing FDG final/latest: ${FDG_BEST:-<unset>}" >&2; exit 1; }

PARENT="${PARENT_STAMP:-${TASK1_NNUNET_RESULTS_STAMP_NAME:-}}"
[[ -n "${PARENT}" ]] || {
  PARENT="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_nnunet_psma_fs${FEWSHOT_N}_f${FOLD_ID}_1gpu_bs${BATCH}_tr${TRAIN_ITERS}_val${VAL_ITERS}e${VAL_EVERY}_${TOTAL_EPOCHS}ep"
}
# If PARENT already ends with _fK, use as stamp; else append
if [[ "${PARENT}" == *_f${FOLD_ID} ]]; then
  STAMP="${PARENT}"
  PARENT="${PARENT%_f${FOLD_ID}}"
else
  STAMP="${PARENT}_f${FOLD_ID}"
fi

if [[ -n "${TASK1_SPLITS_FINAL_JSON:-}" ]]; then
  SPLITS="${TASK1_SPLITS_FINAL_JSON}"
else
  SPLITS="${SPLIT_DIR}/fold${FOLD_ID}_nnunet.json"
fi
[[ -f "${SPLITS}" ]] || { echo "[error] missing ${SPLITS}" >&2; exit 1; }
LOSS_PNG="loss_curve_iclr2026_nnunet_psma_fs${FEWSHOT_N}_f${FOLD_ID}_${STAMP}.png"
LOG="${ICLR_VIS}/nohup_nnunet_psma_fs${FEWSHOT_N}_fold${FOLD_ID}_${STAMP}.log"

echo "[nnunet-fs${FEWSHOT_N}-1f] STAMP=${STAMP} fold=${FOLD_ID} gpu=${GPU_ID} ep=${TOTAL_EPOCHS} bs=${BATCH} tr=${TRAIN_ITERS} val=${VAL_ITERS} every=${VAL_EVERY} best=${BEST_BY} val_loss_only=${VAL_LOSS_ONLY}"

# Guard restart always continues from latest — never pair with pretrained
TASK1_GUARD_STAMP="${STAMP}" \
TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" \
TASK1_GUARD_TRAINER_FOLDER="${TF}" \
TASK1_GUARD_DATASET_DIR="${DS}" \
TASK1_GUARD_TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
TASK1_GUARD_RESTART_SCRIPT="ICLR2026/run/run_nnunet_psma_fewshot50_onefold_bg.sh" \
TASK1_GUARD_REQUIRE_ARM=1 \
TASK1_GUARD_EXTRA_ENV="FOLD_ID=${FOLD_ID},GPU_ID=${GPU_ID},PARENT_STAMP=${PARENT},TASK1_FEWSHOT_N=${FEWSHOT_N},TASK1_FEWSHOT_SPLIT_DIR=${SPLIT_DIR},TASK1_UDA_FDG_BEST=${FDG_BEST},TASK1_UDA_FDG_STAMP=${FDG_STAMP},TASK1_NUM_EPOCHS=${TOTAL_EPOCHS},TASK1_TRAIN_ITERS_PER_EPOCH=${TRAIN_ITERS},TASK1_VAL_ITERS_PER_EPOCH=${VAL_ITERS},TASK1_VAL_EVERY_N_EPOCHS=${VAL_EVERY},TASK1_FS50_VAL_ITERS=${VAL_ITERS},TASK1_FS50_VAL_EVERY_N_EPOCHS=${VAL_EVERY},TASK1_FIXED_BATCH_3D_FULLRES=${BATCH},TASK1_SPLITS_FINAL_JSON=${SPLITS},TASK1_BEST_BY=${BEST_BY},TASK1_VAL_LOSS_ONLY=${VAL_LOSS_ONLY},TASK1_TRAIN_NUM_GPUS=1,TASK1_DOCKER_GPUS=device=${GPU_ID},TASK1_CUDA_VISIBLE_DEVICES=${GPU_ID},FOLD=0,DATASET_ID=${DATASET_ID},TASK1_NNUNET_RESULTS_STAMP_NAME=${STAMP},TASK1_CONTINUE_TRAINING=1,TASK1_CONTINUE_FROM_LATEST=1,TASK1_CONTINUE_PICK_NEWER=1" \
FOLD=0 \
  bash "${ROOT}/run_task/run_task1_train_auto_resume_guard_bg.sh" || true

TASK1_CRASH_MONITOR_STAGE="nnunet_fs${FEWSHOT_N}_f${FOLD_ID}_start" \
TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" \
  bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true

export DATASET_ID TRAINER PLANS_ID CONFIG
export FOLD=0
export TASK1_DATASET_NAME="${DS}"
export TASK1_NUM_EPOCHS="${TOTAL_EPOCHS}"
export TASK1_LR_SCHEDULE_NUM_EPOCHS="${TOTAL_EPOCHS}"
export TASK1_TRAIN_ITERS_PER_EPOCH="${TRAIN_ITERS}"
export TASK1_VAL_ITERS_PER_EPOCH="${VAL_ITERS}"
export TASK1_VAL_EVERY_N_EPOCHS="${VAL_EVERY}"
export TASK1_FS50_VAL_ITERS="${VAL_ITERS}"
export TASK1_FS50_VAL_EVERY_N_EPOCHS="${VAL_EVERY}"
export TASK1_FIXED_BATCH_3D_FULLRES="${BATCH}"
export TASK1_N_PROC_DA="${TASK1_N_PROC_DA:-4}"
export TASK1_SPLITS_FINAL_JSON="${SPLITS}"
# Allow crash-resume / external override to continue from latest
export TASK1_CONTINUE_TRAINING="${TASK1_CONTINUE_TRAINING:-0}"
export TASK1_CONTINUE_FROM_LATEST="${TASK1_CONTINUE_FROM_LATEST:-0}"
export TASK1_CONTINUE_FROM_BEST="${TASK1_CONTINUE_FROM_BEST:-0}"
export TASK1_CONTINUE_PICK_NEWER="${TASK1_CONTINUE_PICK_NEWER:-0}"
# continue 续训时不可再带 pretrained（与 --c 冲突）
if [[ "${TASK1_CONTINUE_TRAINING}" == "1" ]]; then
  unset TASK1_PRETRAINED_WEIGHTS || true
else
  export TASK1_PRETRAINED_WEIGHTS="${FDG_BEST}"
fi
export TASK1_BEST_BY="${BEST_BY}"
export TASK1_VAL_LOSS_ONLY="${VAL_LOSS_ONLY}"
export TASK1_PSMA_VAL_ENABLE=0
# val_loss 选 best 时不画 ema 伪 Dice
export TASK1_LOSS_PLOT_VAL_EMA="${TASK1_LOSS_PLOT_VAL_EMA:-0}"
export TASK1_LOSS_PLOT_VAL_FROM_EPOCH=
export TASK1_SEGMENT_CHECKPOINT=0
export TASK1_PERIODIC_CHECKPOINT_EVERY=0
export TASK1_ALWAYS_SAVE_LATEST=1
export TASK1_SAVE_LATEST_EVERY="${TASK1_SAVE_LATEST_EVERY:-25}"
export TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH=0
export TASK1_RAM_SHARD_ENABLE=0
export TASK1_SKIP_RAW_PREP=1
export TASK1_SKIP_PLAN_PREP=1
export TASK1_FORCE_PLAN_PREP=0
export TASK1_STOP_AFTER_PREP=0
export TASK1_NNUNET_RESULTS_STAMP_SUBDIR=1
export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_TRAIN_NUM_GPUS=1
export TASK1_DOCKER_GPUS="device=${GPU_ID}"
export TASK1_CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TASK1_PREFLIGHT_GPUS="${GPU_ID}"
export TASK1_DOCKER_SHM="${TASK1_DOCKER_SHM:-16g}"
export TASK1_DOCKER_BACKGROUND="${TASK1_DOCKER_BACKGROUND:-1}"
export TASK1_TRAINER_PY="${ROOT}/nnunet_ext_trainers/nnUNetTrainer_Task1StdTrainVal50.py"
export TASK1_INITIAL_LR="${TASK1_INITIAL_LR:-0.001}"
export TASK1_LOSS_OUT_DIR="${ICLR_VIS}"
export TASK1_LOSS_OUT_DIR_EXTRA=none
export TASK1_LOSS_OUT_NAME="${LOSS_PNG}"
export TASK1_LIVE_LOSS_PLOT=1
export TASK1_LOSS_MERGE_ALL_LOGS=1
export TASK1_LOSS_PLOT_X_FOLLOW=1
export TASK1_LOSS_PLOT_SHOW_ETA=1
export TASK1_LOSS_PLOT_SEED_EMPTY=1
export TASK1_LOSS_PLOT_X_MAX_EPOCHS="${TOTAL_EPOCHS}"
# for guard restart
export FOLD_ID GPU_ID PARENT_STAMP="${PARENT}"

TASK1_PREFLIGHT_STEP="nnunet-fs${FEWSHOT_N}-f${FOLD_ID}" \
  bash "${ROOT}/scripts/task1_gpu_train_preflight.sh" || true

bash "${ROOT}/other/task1_train_nnunet_from_dataset1.sh" 2>&1 | tee -a "${LOG}"

sleep 20
TASK1_CRASH_MONITOR_STAGE="nnunet_fs${FEWSHOT_N}_f${FOLD_ID}_running" \
TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" \
TASK1_CRASH_MONITOR_ARM_SEC="${TASK1_CRASH_MONITOR_ARM_SEC:-86400}" \
  bash "${ROOT}/scripts/task1_crash_monitor_arm.sh" || true

echo "[nnunet-fs${FEWSHOT_N}-1f] launched STAMP=${STAMP} log=${LOG}"
