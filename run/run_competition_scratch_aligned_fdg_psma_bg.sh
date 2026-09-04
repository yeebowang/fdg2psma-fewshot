#!/usr/bin/env bash
# Competition *_scratch aligned cascade: FDG (tr70/val0 · 169ep) → PSMA fs50 f258 → TEST20.
# Does NOT use Dataset619 / LesionTracer final / EDT / LocalEdit.
#
# Backbone (interim): Dataset228 + nnUNetTrainer_Task1StdTrainVal50 (same aligned protocol as nnunet).
# MultiTalent organ dual-head wiring is deferred; board note records this.
#
#   TASK1_BOARD_METHOD=hemingduo_scratch bash ICLR2026/run/run_competition_scratch_aligned_fdg_psma_bg.sh
#   TASK1_BOARD_METHOD=chenyixin_scratch bash ICLR2026/run/run_competition_scratch_aligned_fdg_psma_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
ICLR_VIS="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
mkdir -p "${ICLR_VIS}"

METHOD="${TASK1_BOARD_METHOD:-}"
case "${METHOD}" in
  hemingduo_scratch|chenyixin_scratch) ;;
  *)
    echo "[error] set TASK1_BOARD_METHOD=hemingduo_scratch|chenyixin_scratch (got '${METHOD}')" >&2
    exit 2
    ;;
esac

# Refuse accidental final-submit / pretrained init for scratch rows
python3 "${CTRL}/ICLR2026/scripts/assert_competition_board_weights.py" \
  ${TASK1_UDA_FDG_BEST:+"${TASK1_UDA_FDG_BEST}"} ${TASK1_PRETRAINED_WEIGHTS:+"${TASK1_PRETRAINED_WEIGHTS}"} || exit 1
unset TASK1_PRETRAINED_WEIGHTS TASK1_UDA_PRETRAINED || true

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"
export TASK1_BOARD_METHOD="${METHOD}"

FDG_EP="${TASK1_NNUNET_FDG_EPOCHS:-169}"
FDG_TR="${TASK1_TRAIN_ITERS_PER_EPOCH:-70}"
FDG_VAL="${TASK1_NNUNET_FDG_VAL_ITERS:-0}"
PSMA_EP="${TASK1_NNUNET_PSMA_EPOCHS:-${TASK1_NUM_EPOCHS:-100}}"
PSMA_TR="${TASK1_NNUNET_PSMA_TRAIN_ITERS:-25}"
PSMA_VAL="${TASK1_NNUNET_PSMA_VAL_ITERS:-25}"
PSMA_EVERY="${TASK1_NNUNET_PSMA_VAL_EVERY:-20}"
PSMA_BS="${TASK1_FIXED_BATCH_3D_FULLRES:-2}"

PIPE_LOG="${ICLR_VIS}/nohup_${METHOD}_aligned_fdg${FDG_EP}_psma_f258.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[${METHOD}] FDG ${FDG_EP}ep tr${FDG_TR}/val${FDG_VAL} → PSMA tr${PSMA_TR}/val${PSMA_VAL}e${PSMA_EVERY} ${PSMA_EP}ep bs${PSMA_BS}"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"${METHOD}\":{\"fdg_pretrain\":{\"status\":\"running\",\"total_epochs\":${FDG_EP},\"train_iters\":${FDG_TR},\"val_iters\":${FDG_VAL},\"bs\":6,\"bs_note\":\"gbs 3GPU\",\"note\":\"scratch FDG tr${FDG_TR}/val${FDG_VAL} · ${FDG_EP}ep (aligned nnUNet backbone; no Dataset619)\",\"train_sec\":null,\"train_time\":null},\"psma_fs50_f258\":{\"status\":\"queued\",\"stamp\":\"\",\"total_epochs\":${PSMA_EP},\"train_iters\":${PSMA_TR},\"val_iters\":${PSMA_VAL},\"online_val\":\"VAL${PSMA_VAL} every${PSMA_EVERY}\",\"bs\":${PSMA_BS},\"metric\":\"TEST20 Dice; best=min val_loss\",\"note\":\"queued after FDG\",\"epoch\":null,\"train_sec\":null,\"train_time\":null,\"fold_dice\":{},\"mean\":null,\"fold_ckpt_ep\":{},\"test_invalidated\":true}}},\"queue\":[\"${METHOD}.fdg tr${FDG_TR}/val${FDG_VAL} ${FDG_EP}ep\",\"${METHOD}.psma f258\",\"${METHOD}.test20\"],\"updated_note\":\"${METHOD} aligned FDG running\"}" || true

# --- FDG ---
export TASK1_NUM_EPOCHS="${FDG_EP}"
export TASK1_TRAIN_ITERS_PER_EPOCH="${FDG_TR}"
export TASK1_VAL_ITERS_PER_EPOCH="${FDG_VAL}"
export TASK1_FIXED_BATCH_3D_FULLRES="${TASK1_NNUNET_FDG_BATCH:-6}"
export TASK1_BEST_BY=train_loss
export TASK1_VAL_LOSS_ONLY=0
export TASK1_PSMA_VAL_ENABLE=0
export TASK1_VAL_ITERS_LATE_FROM_EPOCH=999999
export TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH=0
export TASK1_LOSS_OUT_NAME="loss_curve_iclr2026_${METHOD}_fdg.png"
unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

bash "${CTRL}/ICLR2026/run/run_baseline1_fdg_2ch_fullres_3000ep_bg.sh"

FDG_STAMP=""
if [[ -f "${ICLR_VIS}/${METHOD}_fdg_LAST_STAMP.txt" ]]; then
  FDG_STAMP="$(tr -d '[:space:]' < "${ICLR_VIS}/${METHOD}_fdg_LAST_STAMP.txt")"
fi
if [[ -z "${FDG_STAMP}" ]]; then
  FDG_STAMP="$(ls -1dt "${WORK}/nnUNet_results/"*_iclr2026_${METHOD}_fdg_*_${FDG_EP}ep* 2>/dev/null | head -1 | xargs -I{} basename {} || true)"
fi
[[ -n "${FDG_STAMP}" ]] || { echo "[error] cannot resolve ${METHOD} FDG stamp" >&2; exit 1; }
echo "${FDG_STAMP}" > "${ICLR_VIS}/${METHOD}_fdg_LAST_STAMP.txt"
FDG_FOLD="${WORK}/nnUNet_results/${FDG_STAMP}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0"
FDG_FINAL="${FDG_FOLD}/checkpoint_final.pth"

_resolve_fdg_ckpt() {
  local f
  for f in checkpoint_final.pth checkpoint_latest.pth; do
    if [[ -f "${FDG_FOLD}/${f}" ]]; then
      echo "${FDG_FOLD}/${f}"
      return 0
    fi
  done
  return 1
}

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"${METHOD}\":{\"fdg_pretrain\":{\"status\":\"running\",\"stamp\":\"${FDG_STAMP}\",\"total_epochs\":${FDG_EP},\"train_iters\":${FDG_TR},\"val_iters\":${FDG_VAL},\"bs\":6,\"note\":\"scratch FDG · ${FDG_STAMP}\"}}},\"updated_note\":\"${METHOD} FDG ${FDG_STAMP}\"}" || true

echo "[${METHOD}] wait FDG ${FDG_STAMP}…"
while [[ ! -f "${FDG_FINAL}" ]]; do
  if FDG_CKPT="$(_resolve_fdg_ckpt)" && ! docker ps --format '{{.Names}}' | grep -qiE "baseline1|${METHOD}|fdg"; then
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
FDG_CKPT="$(_resolve_fdg_ckpt)" || true
[[ -n "${FDG_CKPT}" && -f "${FDG_CKPT}" ]] || { echo "[error] missing FDG ckpt under ${FDG_FOLD}" >&2; exit 1; }
echo "[${METHOD}] FDG done; PSMA init=${FDG_CKPT}"

# stage boundary: disarm FDG crash-monitor before PSMA
TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}" bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"${METHOD}\":{\"fdg_pretrain\":{\"status\":\"done\",\"stamp\":\"${FDG_STAMP}\",\"best_ckpt\":\"${FDG_CKPT}\",\"total_epochs\":${FDG_EP},\"note\":\"scratch FDG done · ${FDG_EP}ep\"},\"psma_fs50_f258\":{\"status\":\"running\",\"note\":\"PSMA after scratch FDG\"}}},\"updated_note\":\"${METHOD} FDG done → PSMA\"}" || true

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
export TASK1_BOARD_METHOD="${METHOD}"
unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

bash "${CTRL}/ICLR2026/run/run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh"

echo "[${METHOD}] ALL DONE"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"updated_note\":\"${METHOD} aligned FDG+PSMA+TEST done\"}" || true
