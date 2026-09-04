#!/usr/bin/env bash
# nnUNet PSMA f5: restore ep100 seed → short retrain to ep120 (ema_fg_dice) → TEST20 f5.
#
#   bash ICLR2026/run/run_nnunet_f5_dice_best_test20_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"

PARENT="${NN_PARENT:-20260818_010404_iclr2026_nnunet_psma_fs50_f258_1gpu_bs2_tr25_val25e20_100ep_gpu013}"
FOLD=5
GPU="${TASK1_GPU_ID:-1}"
STAMP="${PARENT}_f${FOLD}"
FD="${WORK}/nnUNet_results/${STAMP}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0"
SEED_BAK="${FD}/checkpoint_final.pth.bak_before_best_20260818_094018"
PIPE_LOG="${VIS}/nohup_nnunet_f5_dice_best_test20.log"

exec > >(tee -a "${PIPE_LOG}") 2>&1
echo "[nn-f5] PARENT=${PARENT} dice-best short retrain ep100→120"

[[ -f "${SEED_BAK}" ]] || { echo "[error] missing seed ${SEED_BAK}" >&2; exit 1; }

docker run --rm --user root --entrypoint bash -v "${DATA}:${DATA}" autopet_baseline:latest -lc \
  "set -e; cp -a '${SEED_BAK}' '${FD}/checkpoint_latest.pth'; cp -a '${SEED_BAK}' '${FD}/checkpoint_best.pth'; echo restored_ep100_seed"

rm -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${STAMP}.txt"

FOLD_ID="${FOLD}" GPU_ID="${GPU}" PARENT_STAMP="${PARENT}" \
  TASK1_NUM_EPOCHS=120 \
  TASK1_LR_SCHEDULE_NUM_EPOCHS=100 \
  TASK1_TRAIN_ITERS_PER_EPOCH=25 \
  TASK1_VAL_ITERS_PER_EPOCH=25 \
  TASK1_FS50_VAL_ITERS=25 \
  TASK1_VAL_EVERY_N_EPOCHS=20 \
  TASK1_FS50_VAL_EVERY_N_EPOCHS=20 \
  TASK1_FIXED_BATCH_3D_FULLRES=2 \
  TASK1_BEST_BY=ema_fg_dice \
  TASK1_VAL_LOSS_ONLY=0 \
  TASK1_CONTINUE_TRAINING=1 \
  TASK1_CONTINUE_FROM_LATEST=1 \
  TASK1_CONTINUE_FROM_BEST=0 \
  TASK1_CONTINUE_PICK_NEWER=1 \
  bash "${CTRL}/ICLR2026/run/run_nnunet_psma_fewshot50_onefold_bg.sh"

echo "[nn-f5] waiting training STAMP=${STAMP}"
while true; do
  if [[ -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${STAMP}.txt" ]]; then
    echo "[nn-f5] stopped by flag"
    break
  fi
  if ! pgrep -af "${STAMP}" >/dev/null 2>&1 \
     && ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qF "${STAMP}"; then
    echo "[nn-f5] container exited"
    break
  fi
  sleep 45
done
TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

python3 << PY
from pathlib import Path
import sys
sys.path.insert(0, "${CTRL}/ICLR2026/scripts")
from nnunet_pseudo_dice_best import pseudo_dice_best_epoch
fd = Path("${FD}")
ep, dice, _ = pseudo_dice_best_epoch(fd)
print(f"[nn-f5] pseudo_dice best ep={ep} dice={dice}")
PY

echo "[nn-f5] TEST20 fold5 only"
rm -f "${WORK}/nnUNet_results/${PARENT}/psma_test20_eval/fold5/score_detail.json"
PARENT_STAMP="${PARENT}" TASK1_FOLDS=5 TASK1_TEST_SKIP_DONE=0 \
  bash "${CTRL}/ICLR2026/run/run_nnunet_psma_test20_f258_parallel.sh"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" \
  --patch-json '{"updated_note":"nnUNet f5 dice-best@120 TEST20 refreshed"}' || true

echo "[nn-f5] chain SegAnyPET f5 resume (GPU1 free)"
env SEG_RESUME_FOLDS=5 bash "${CTRL}/ICLR2026/run/run_seganypet_resume_until_decline_bg.sh" || true

echo "[nn-f5] DONE log=${PIPE_LOG}"
