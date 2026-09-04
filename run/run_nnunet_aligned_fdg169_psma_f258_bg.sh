#!/usr/bin/env bash
# nnUNet 全流程对齐 DpDNet：
#   FDG: tr70/val0 · 169ep · bs=6 (3GPU) · best=min train_loss（无 online val）
#   PSMA fs50 f258: tr25/val25 every20 · 100ep · bs=2 · 1fold/GPU · best=val_loss → TEST20
#
#   bash ICLR2026/run/run_nnunet_aligned_fdg169_psma_f258_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
ICLR_VIS="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
mkdir -p "${ICLR_VIS}"

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"

FDG_EP="${TASK1_NNUNET_FDG_EPOCHS:-169}"
FDG_TR="${TASK1_TRAIN_ITERS_PER_EPOCH:-70}"
FDG_VAL="${TASK1_NNUNET_FDG_VAL_ITERS:-0}"
PSMA_EP="${TASK1_NNUNET_PSMA_EPOCHS:-${TASK1_NUM_EPOCHS:-100}}"
PSMA_TR="${TASK1_NNUNET_PSMA_TRAIN_ITERS:-25}"
PSMA_VAL="${TASK1_NNUNET_PSMA_VAL_ITERS:-25}"
PSMA_EVERY="${TASK1_NNUNET_PSMA_VAL_EVERY:-20}"
PSMA_BS="${TASK1_FIXED_BATCH_3D_FULLRES:-2}"

PIPE_LOG="${ICLR_VIS}/nohup_nnunet_aligned_fdg${FDG_EP}_psma_f258.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[nnunet-aligned] FDG ${FDG_EP}ep tr${FDG_TR}/val${FDG_VAL} → PSMA tr${PSMA_TR}/val${PSMA_VAL}e${PSMA_EVERY} ${PSMA_EP}ep bs${PSMA_BS}"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"nnunet\":{\"fdg_pretrain\":{\"status\":\"running\",\"total_epochs\":${FDG_EP},\"train_iters\":${FDG_TR},\"val_iters\":${FDG_VAL},\"bs\":6,\"note\":\"aligned to DpDNet · tr${FDG_TR}/val${FDG_VAL} · ${FDG_EP}ep\",\"train_sec\":null,\"train_time\":null},\"psma_fs50_f258\":{\"status\":\"queued\",\"stamp\":\"\",\"total_epochs\":${PSMA_EP},\"train_iters\":${PSMA_TR},\"val_iters\":${PSMA_VAL},\"online_val\":\"VAL${PSMA_VAL} every${PSMA_EVERY}\",\"bs\":${PSMA_BS},\"metric\":\"TEST20 Dice; best=min val_loss\",\"note\":\"queued · match DpDNet PSMA\",\"epoch\":null,\"train_sec\":null,\"train_time\":null,\"fold_dice\":{},\"mean\":null,\"fold_ckpt_ep\":{},\"test_invalidated\":true}}},\"queue\":[\"nnunet.fdg tr${FDG_TR}/val${FDG_VAL} ${FDG_EP}ep\",\"nnunet.psma f258\",\"nnunet.test20\"],\"updated_note\":\"nnUNet aligned pipeline running FDG tr${FDG_TR}/val${FDG_VAL} ${FDG_EP}ep\"}" || true

# --- FDG (match DpDNet: no online val) ---
export TASK1_NUM_EPOCHS="${FDG_EP}"
export TASK1_TRAIN_ITERS_PER_EPOCH="${FDG_TR}"
export TASK1_VAL_ITERS_PER_EPOCH="${FDG_VAL}"
# FDG: 3GPU DDP needs global batch >= n_gpu (default 6)
export TASK1_FIXED_BATCH_3D_FULLRES="${TASK1_NNUNET_FDG_BATCH:-6}"
# val=0 → best must be train_loss (not val_loss)
export TASK1_BEST_BY=train_loss
export TASK1_VAL_LOSS_ONLY=0
export TASK1_PSMA_VAL_ENABLE=0
export TASK1_VAL_ITERS_LATE_FROM_EPOCH=999999
export TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH=0
unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

bash "${CTRL}/ICLR2026/run/run_baseline1_fdg_2ch_fullres_3000ep_bg.sh"

# resolve FDG stamp / best
FDG_STAMP=""
if [[ -f "${ICLR_VIS}/baseline1_fdg_LAST_STAMP.txt" ]]; then
  FDG_STAMP="$(tr -d '[:space:]' < "${ICLR_VIS}/baseline1_fdg_LAST_STAMP.txt")"
fi
if [[ -z "${FDG_STAMP}" ]]; then
  FDG_STAMP="$(ls -1dt "${WORK}/nnUNet_results/"*_iclr2026_baseline1_fdg_*_${FDG_EP}ep* 2>/dev/null | head -1 | xargs -I{} basename {} || true)"
fi
[[ -n "${FDG_STAMP}" ]] || { echo "[error] cannot resolve FDG stamp" >&2; exit 1; }
echo "${FDG_STAMP}" > "${ICLR_VIS}/baseline1_fdg_LAST_STAMP.txt"
FDG_FOLD="${WORK}/nnUNet_results/${FDG_STAMP}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0"
FDG_FINAL="${FDG_FOLD}/checkpoint_final.pth"
FDG_CKPT=""
_resolve_nn_fdg_ckpt() {
  local f
  for f in checkpoint_final.pth checkpoint_latest.pth; do
    if [[ -f "${FDG_FOLD}/${f}" ]]; then
      echo "${FDG_FOLD}/${f}"
      return 0
    fi
  done
  return 1
}

# stamp must be on the board while FDG is running (otherwise ETA shows "starting…")
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"nnunet\":{\"fdg_pretrain\":{\"status\":\"running\",\"stamp\":\"${FDG_STAMP}\",\"total_epochs\":${FDG_EP},\"train_iters\":${FDG_TR},\"val_iters\":${FDG_VAL},\"bs\":6,\"note\":\"aligned to DpDNet · tr${FDG_TR}/val${FDG_VAL} · ${FDG_EP}ep\"}}},\"updated_note\":\"nnUNet FDG running · ${FDG_STAMP}\"}" || true

# NOTE: do not use unquoted '>' in echo (redirect). Prefer final then latest.
echo "[nnunet-aligned] wait FDG ${FDG_STAMP} (prefer checkpoint_final, else latest)…"
while [[ ! -f "${FDG_FINAL}" ]]; do
  if FDG_CKPT="$(_resolve_nn_fdg_ckpt)" && ! docker ps --format '{{.Names}}' | grep -qi baseline1; then
    ep="$(
      python3 -c "
from pathlib import Path
import re
fd=Path(r'''${FDG_FOLD}''')
logs=sorted(fd.glob('training_log*.txt'), key=lambda p:p.stat().st_mtime, reverse=True)
ep=0
if logs:
  for line in logs[0].read_text(errors='ignore').splitlines():
    m=re.search(r'Epoch[: ]+(\d+)', line, re.I)
    if m: ep=max(ep,int(m.group(1)))
print(ep)
"
    )"
    [[ "${ep}" -ge "$((FDG_EP - 1))" ]] && break
  fi
  sleep 90
done
FDG_CKPT="$(_resolve_nn_fdg_ckpt)" || true
[[ -n "${FDG_CKPT}" && -f "${FDG_CKPT}" ]] || { echo "[error] missing FDG final/latest under ${FDG_FOLD}" >&2; exit 1; }
echo "[nnunet-aligned] FDG done; PSMA init=${FDG_CKPT}"

# leave FDG crash-monitor before next stage
TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}" bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"nnunet\":{\"fdg_pretrain\":{\"status\":\"done\",\"stamp\":\"${FDG_STAMP}\",\"best_ckpt\":\"${FDG_CKPT}\",\"total_epochs\":${FDG_EP},\"note\":\"aligned FDG done · ${FDG_EP}ep · PSMA init=final/latest\"},\"psma_fs50_f258\":{\"status\":\"running\",\"note\":\"aligned PSMA after FDG (init=final/latest)\"}}},\"updated_note\":\"nnUNet FDG ${FDG_EP}ep done -> PSMA\",\"queue\":[\"nnunet.psma f258\",\"nnunet.test20\"]}" || true

# --- PSMA + TEST ---
export TASK1_UDA_FDG_STAMP="${FDG_STAMP}"
export TASK1_UDA_FDG_BEST="${FDG_CKPT}"
export TASK1_NUM_EPOCHS="${PSMA_EP}"
export TASK1_TRAIN_ITERS_PER_EPOCH="${PSMA_TR}"
export TASK1_FS50_VAL_ITERS="${PSMA_VAL}"
export TASK1_FS50_VAL_EVERY_N_EPOCHS="${PSMA_EVERY}"
export TASK1_VAL_EVERY_N_EPOCHS="${PSMA_EVERY}"
export TASK1_VAL_ITERS_PER_EPOCH="${PSMA_VAL}"
export TASK1_FIXED_BATCH_3D_FULLRES="${PSMA_BS}"
export TASK1_BEST_BY=val_loss
export TASK1_VAL_LOSS_ONLY=1
export TASK1_FOLDS=2,5,8
export TASK1_FOLD_GPUS=2:0,5:1,8:3
export TASK1_TEST_SKIP_DONE=1
unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

bash "${CTRL}/ICLR2026/run/run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh"

echo "[nnunet-aligned] ALL DONE"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json '{"queue":[],"updated_note":"nnUNet aligned FDG+PSMA+TEST done"}' || true
