#!/usr/bin/env bash
# nnUNet PSMA fc70%: single run · tr25 val25 bs2 · 100ep → decline → TEST20
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/ICLR2026/run/_psma_fc70_env.sh"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
MON="${ROOT}/ICLR2026/scripts/monitor_val_dice_decline_stop.py"

read -r TR VAL NTR NVAL < <(_fc70_resolve_iters)
FDG_STAMP="${TASK1_UDA_FDG_STAMP:-20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep}"
FDG_BEST="${TASK1_UDA_FDG_BEST:-/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth}"

PARENT="${TASK1_NNUNET_RESULTS_STAMP_NAME:-}"
if [[ -z "${PARENT}" || "${PARENT}" != *fc70* ]]; then
  PARENT="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_nnunet_psma_fc70_1gpu_bs${PSMA_FC70_BS}_tr${TR}_val${VAL}e${PSMA_FC70_VAL_EVERY}_${PSMA_FC70_EP}ep_gpu${PSMA_FC70_GPU}"
fi

PIPE_LOG="${VIS}/nohup_nnunet_psma_fc70_${PARENT}.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[nnunet-fc70] PARENT=${PARENT} n=${NTR}/${NVAL} tr${TR}/val${VAL}e${PSMA_FC70_VAL_EVERY} init=${FDG_BEST}"

BOARD_METHOD="${TASK1_BOARD_METHOD:-nnunet}"
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"${PSMA_FC70_STAGE}\":{\"status\":\"running\",\"stamp\":\"${PARENT}\",\"bs\":${PSMA_FC70_BS},\"train_iters\":${TR},\"val_iters\":${VAL},\"total_epochs\":${PSMA_FC70_EP},\"online_val\":\"VAL${VAL} every${PSMA_FC70_VAL_EVERY}\",\"note\":\"fc70% PSMA · single run · n=${NTR}/${NVAL}\",\"test_invalidated\":true}}},\"updated_note\":\"${BOARD_METHOD} fc70 training\"}" || true

bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true

FOLD_ID="${PSMA_FC70_FOLD}" GPU_ID="${PSMA_FC70_GPU}" PARENT_STAMP="${PARENT}" \
  TASK1_FEWSHOT_N=70 TASK1_PSMA_BOARD_STAGE="${PSMA_FC70_STAGE}" \
  TASK1_FEWSHOT_SPLIT_DIR="${ROOT}/ICLR2026/data" \
  TASK1_SPLITS_FINAL_JSON="${PSMA_FC70_SPLITS}" \
  TASK1_UDA_FDG_STAMP="${FDG_STAMP}" TASK1_UDA_FDG_BEST="${FDG_BEST}" \
  TASK1_NUM_EPOCHS="${PSMA_FC70_EP}" \
  TASK1_TRAIN_ITERS_PER_EPOCH="${TR}" \
  TASK1_VAL_ITERS_PER_EPOCH="${VAL}" \
  TASK1_FS50_VAL_ITERS="${VAL}" \
  TASK1_VAL_EVERY_N_EPOCHS="${PSMA_FC70_VAL_EVERY}" \
  TASK1_FS50_VAL_EVERY_N_EPOCHS="${PSMA_FC70_VAL_EVERY}" \
  TASK1_FIXED_BATCH_3D_FULLRES="${PSMA_FC70_BS}" \
  TASK1_BEST_BY=val_loss TASK1_VAL_LOSS_ONLY=1 \
  bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fewshot50_onefold_bg.sh"

STAMP="${PARENT}_f${PSMA_FC70_FOLD}"
rm -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${STAMP}.txt"
nohup python3 "${MON}" --method nnunet --parent-stamp "${PARENT}" --fold "${PSMA_FC70_FOLD}" \
  --base-ep "${PSMA_FC70_BASE_EP}" --val-every "${PSMA_FC70_VAL_EVERY}" \
  >"${VIS}/nohup_decline_mon_nnunet_fc70.log" 2>&1 &

FOLD_ID="${PSMA_FC70_FOLD}" GPU_ID="${PSMA_FC70_GPU}" PARENT_STAMP="${PARENT}" \
  TASK1_FEWSHOT_N=70 TASK1_FEWSHOT_SPLIT_DIR="${ROOT}/ICLR2026/data" \
  TASK1_SPLITS_FINAL_JSON="${PSMA_FC70_SPLITS}" \
  TASK1_UDA_FDG_STAMP="${FDG_STAMP}" TASK1_UDA_FDG_BEST="${FDG_BEST}" \
  TASK1_NUM_EPOCHS="${PSMA_FC70_MAX_EP}" TASK1_LR_SCHEDULE_NUM_EPOCHS="${PSMA_FC70_BASE_EP}" \
  TASK1_TRAIN_ITERS_PER_EPOCH="${TR}" TASK1_VAL_ITERS_PER_EPOCH="${VAL}" \
  TASK1_FS50_VAL_ITERS="${VAL}" TASK1_VAL_EVERY_N_EPOCHS="${PSMA_FC70_VAL_EVERY}" \
  TASK1_FS50_VAL_EVERY_N_EPOCHS="${PSMA_FC70_VAL_EVERY}" \
  TASK1_FIXED_BATCH_3D_FULLRES="${PSMA_FC70_BS}" \
  TASK1_BEST_BY=ema_fg_dice TASK1_VAL_LOSS_ONLY=0 \
  TASK1_CONTINUE_TRAINING=1 TASK1_CONTINUE_FROM_BEST=1 \
  bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fewshot50_onefold_bg.sh"

while [[ ! -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${STAMP}.txt" ]]; do
  pgrep -af "${STAMP}" >/dev/null 2>&1 || docker ps --format '{{.Names}}' 2>/dev/null | grep -qF "${STAMP}" || break
  sleep 60
done
TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true

PARENT_STAMP="${PARENT}" TASK1_PSMA_FC70_FOLD="${PSMA_FC70_FOLD}" \
  bash "${ROOT}/ICLR2026/run/run_nnunet_psma_test20_fc70_bg.sh"

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" --no-plot || true

echo "[nnunet-fc70] ALL DONE ${PARENT}"
