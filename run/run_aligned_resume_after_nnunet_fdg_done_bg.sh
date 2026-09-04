#!/usr/bin/env bash
# Resume after FDG MAE-fullcase already finished:
#   nnUNet PSMA f258 + TEST20 → DpDNet MAE-fullcase
#   bash ICLR2026/run/run_aligned_resume_after_nnunet_fdg_done_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
ICLR_VIS="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
mkdir -p "${ICLR_VIS}"

FDG_STAMP="${TASK1_UDA_FDG_STAMP:-}"
if [[ -z "${FDG_STAMP}" && -f "${ICLR_VIS}/baseline1_fdg_LAST_STAMP.txt" ]]; then
  FDG_STAMP="$(tr -d '[:space:]' < "${ICLR_VIS}/baseline1_fdg_LAST_STAMP.txt")"
fi
[[ -n "${FDG_STAMP}" ]] || { echo "[error] no FDG stamp" >&2; exit 1; }

FDG_FOLD="${WORK}/nnUNet_results/${FDG_STAMP}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0"
FDG_CKPT=""
for f in checkpoint_best.pth checkpoint_final.pth checkpoint_latest.pth; do
  if [[ -f "${FDG_FOLD}/${f}" ]]; then
    FDG_CKPT="${FDG_FOLD}/${f}"
    break
  fi
done
[[ -n "${FDG_CKPT}" && -f "${FDG_CKPT}" ]] || { echo "[error] no FDG ckpt under ${FDG_FOLD}" >&2; exit 1; }

PSMA_EP="${TASK1_NNUNET_PSMA_EPOCHS:-100}"
# Do NOT inherit FDG leftover TASK1_FIXED_BATCH_3D_FULLRES / iters from parent shell
PSMA_BS="${TASK1_NNUNET_PSMA_BATCH:-2}"
VAL_EVERY="${TASK1_NNUNET_VAL_EVERY:-20}"
PSMA_SPLITS="${CTRL}/ICLR2026/data/splits_mae_psma_fewshot50_9fold/fold2_nnunet.json"
read -r PSMA_TR _PSMA_VAL_COMPUTED < <(
  python3 - <<PY
import json, math
from pathlib import Path
d = json.loads(Path("${PSMA_SPLITS}").read_text())[0]
bs = int("${PSMA_BS}")
print(max(1, len(d["train"]) // bs), max(1, math.ceil(len(d["val"]) / bs)))
PY
)
PSMA_VAL="${TASK1_NNUNET_PSMA_VAL_ITERS:-25}"
echo "[resume] computed PSMA gbs=${PSMA_BS} tr=${PSMA_TR} val=${PSMA_VAL}e${VAL_EVERY}"
# clear polluted FDG schedule before exporting PSMA
unset TASK1_TRAIN_ITERS_PER_EPOCH TASK1_VAL_ITERS_PER_EPOCH TASK1_FS50_VAL_ITERS \
  TASK1_FIXED_BATCH_3D_FULLRES TASK1_NUM_EPOCHS TASK1_VAL_EVERY_N_EPOCHS \
  TASK1_FS50_VAL_EVERY_N_EPOCHS || true

PIPE_LOG="${ICLR_VIS}/nohup_aligned_resume_after_nnunet_fdg_done.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[resume] FDG done stamp=${FDG_STAMP} init=${FDG_CKPT}"
echo "[resume] → nnUNet PSMA tr${PSMA_TR}/val${PSMA_VAL}e${VAL_EVERY} ${PSMA_EP}ep → TEST20 → DpDNet"

# leave FDG crash-monitor before next stage
TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}" bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"nnunet\":{\"fdg_pretrain\":{\"status\":\"done\",\"stamp\":\"${FDG_STAMP}\",\"best_ckpt\":\"${FDG_CKPT}\",\"total_epochs\":100,\"note\":\"MAE-align FDG done · init=$(basename "${FDG_CKPT}")\"},\"psma_fs50_f258\":{\"status\":\"running\",\"note\":\"MAE-align PSMA after FDG (init=best)\",\"total_epochs\":${PSMA_EP},\"train_iters\":${PSMA_TR},\"val_iters\":${PSMA_VAL},\"gbs\":${PSMA_BS},\"bs\":${PSMA_BS},\"test_invalidated\":true,\"fold_dice\":{},\"mean\":null,\"fold_ckpt_ep\":{}}}},\"queue\":[\"nnunet.psma f258\",\"nnunet.test20\",\"dpdnet.fullcase\"],\"updated_note\":\"nnUNet FDG done → PSMA f258\"}" || true

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"
export TASK1_UDA_FDG_STAMP="${FDG_STAMP}"
export TASK1_UDA_FDG_BEST="${FDG_CKPT}"
export TASK1_NUM_EPOCHS="${PSMA_EP}"
export TASK1_TRAIN_ITERS_PER_EPOCH="${PSMA_TR}"
export TASK1_FS50_VAL_ITERS="${PSMA_VAL}"
export TASK1_FS50_VAL_EVERY_N_EPOCHS="${VAL_EVERY}"
export TASK1_VAL_EVERY_N_EPOCHS="${VAL_EVERY}"
export TASK1_VAL_ITERS_PER_EPOCH="${PSMA_VAL}"
export TASK1_FIXED_BATCH_3D_FULLRES="${PSMA_BS}"
export TASK1_BEST_BY=ema_fg_dice
export TASK1_VAL_LOSS_ONLY=0
export TASK1_FOLDS=2,5,8
export TASK1_FOLD_GPUS=2:0,5:1,8:3
export TASK1_TEST_SKIP_DONE=1
unset TASK1_NNUNET_RESULTS_STAMP_NAME || true
export TASK1_CONTINUE_TRAINING=0
export TASK1_CONTINUE_PICK_NEWER=0
export TASK1_CONTINUE_FROM_LATEST=0

bash "${CTRL}/ICLR2026/run/run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh"

echo "[resume] nnUNet PSMA+TEST done → DpDNet MAE-fullcase"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json '{"queue":["dpdnet.fullcase"],"updated_note":"nnUNet MAE-fullcase PSMA+TEST done → DpDNet"}' || true

bash "${CTRL}/ICLR2026/run/run_dpdnet_aligned_fdg_psma_f258_mae_fullcase_bg.sh"

echo "[resume] ALL DONE"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json '{"queue":[],"updated_note":"nnUNet+DpDNet MAE-fullcase ALL done"}' || true
