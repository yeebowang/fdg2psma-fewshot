#!/usr/bin/env bash
# nnUNet + PET/CT MIM init → FDG 169ep (same protocol as scratch nnUNet) → PSMA fs50 f258.
# Init: converted nnunet_v2_mim_best_nnunetformat.pth (not raw MIM wrapper).
#
#   bash ICLR2026/run/run_nnunet_mim_aligned_fdg_psma_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
ICLR_VIS="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
MIM_RAW="${CTRL}/ICLR2026/3D-MAE-PET-CT/weights/nnunetv2/nnunet_v2_mim_best.pth"
MIM_NNUNET="${TASK1_NNUNET_MIM_CKPT:-${CTRL}/ICLR2026/3D-MAE-PET-CT/weights/nnunetv2/nnunet_v2_mim_best_nnunetformat.pth}"
TEMPLATE="${TASK1_NNUNET_MIM_TEMPLATE:-${WORK}/nnUNet_results/20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth}"
mkdir -p "${ICLR_VIS}"

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"
export TASK1_BOARD_METHOD=nnunet_mim

if [[ ! -f "${MIM_NNUNET}" ]]; then
  echo "[nnunet-mim] converting MIM → nnUNet -pretrained_weights format"
  docker run --rm \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    iclr2026_3dmae_petct:cu118 \
    python3 "${CTRL}/ICLR2026/scripts/convert_nnunet_mim_to_pretrained.py" \
      --mim "${MIM_RAW}" --template "${TEMPLATE}" --out "${MIM_NNUNET}"
fi
[[ -f "${MIM_NNUNET}" ]] || { echo "[error] missing converted MIM ckpt ${MIM_NNUNET}" >&2; exit 1; }

FDG_EP="${TASK1_NNUNET_FDG_EPOCHS:-169}"
FDG_TR="${TASK1_TRAIN_ITERS_PER_EPOCH:-70}"
FDG_VAL="${TASK1_NNUNET_FDG_VAL_ITERS:-0}"
PSMA_EP="${TASK1_NNUNET_PSMA_EPOCHS:-${TASK1_NUM_EPOCHS:-100}}"
PSMA_TR="${TASK1_NNUNET_PSMA_TRAIN_ITERS:-25}"
PSMA_VAL="${TASK1_NNUNET_PSMA_VAL_ITERS:-25}"
PSMA_EVERY="${TASK1_NNUNET_PSMA_VAL_EVERY:-20}"
PSMA_BS="${TASK1_FIXED_BATCH_3D_FULLRES:-2}"

PIPE_LOG="${ICLR_VIS}/nohup_nnunet_mim_aligned_fdg${FDG_EP}_psma_f258.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1
echo "[nnunet-mim] FDG ${FDG_EP}ep MIM init → PSMA fs50"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"nnunet_mim\":{\"fdg_pretrain\":{\"status\":\"running\",\"total_epochs\":${FDG_EP},\"train_iters\":${FDG_TR},\"val_iters\":${FDG_VAL},\"bs\":6,\"note\":\"PET+CT MIM → FDG tr${FDG_TR}/val${FDG_VAL} · ${FDG_EP}ep\"}}},\"updated_note\":\"nnUNet MIM FDG running\"}" || true

export TASK1_NUM_EPOCHS="${FDG_EP}"
export TASK1_TRAIN_ITERS_PER_EPOCH="${FDG_TR}"
export TASK1_VAL_ITERS_PER_EPOCH="${FDG_VAL}"
export TASK1_FIXED_BATCH_3D_FULLRES="${TASK1_NNUNET_FDG_BATCH:-6}"
export TASK1_BEST_BY=train_loss
export TASK1_VAL_LOSS_ONLY=0
export TASK1_PSMA_VAL_ENABLE=0
export TASK1_VAL_ITERS_LATE_FROM_EPOCH=999999
export TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH=0
export TASK1_PRETRAINED_WEIGHTS="${MIM_NNUNET}"
export TASK1_CONTINUE_TRAINING=0
unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
bash "${CTRL}/ICLR2026/run/run_baseline1_fdg_2ch_fullres_3000ep_bg.sh"

FDG_STAMP=""
if [[ -f "${ICLR_VIS}/nnunet_mim_fdg_LAST_STAMP.txt" ]]; then
  FDG_STAMP="$(tr -d '[:space:]' < "${ICLR_VIS}/nnunet_mim_fdg_LAST_STAMP.txt")"
fi
if [[ -z "${FDG_STAMP}" ]]; then
  FDG_STAMP="$(ls -1dt "${WORK}/nnUNet_results/"*_iclr2026_nnunet_mim_fdg_* 2>/dev/null | head -1 | xargs -I{} basename {} || true)"
fi
[[ -n "${FDG_STAMP}" ]] || { echo "[error] cannot resolve nnUNet MIM FDG stamp" >&2; exit 1; }
echo "${FDG_STAMP}" > "${ICLR_VIS}/nnunet_mim_fdg_LAST_STAMP.txt"

FDG_FOLD="${WORK}/nnUNet_results/${FDG_STAMP}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0"
FDG_FINAL="${FDG_FOLD}/checkpoint_final.pth"
_resolve_nn_fdg_ckpt() {
  local f
  for f in checkpoint_final.pth checkpoint_latest.pth checkpoint_best.pth; do
    if [[ -f "${FDG_FOLD}/${f}" ]]; then
      echo "${FDG_FOLD}/${f}"
      return 0
    fi
  done
  return 1
}

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"nnunet_mim\":{\"fdg_pretrain\":{\"status\":\"running\",\"stamp\":\"${FDG_STAMP}\",\"total_epochs\":${FDG_EP},\"train_iters\":${FDG_TR},\"val_iters\":${FDG_VAL},\"bs\":6,\"note\":\"PET+CT MIM → FDG tr${FDG_TR}/val${FDG_VAL} · ${FDG_EP}ep\"}}},\"updated_note\":\"nnUNet MIM FDG running · ${FDG_STAMP}\"}" || true

echo "[nnunet-mim] wait FDG ${FDG_STAMP}…"
_fdg_train_alive() {
  local id
  for id in $(docker ps -q 2>/dev/null); do
    if docker inspect "$id" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
        | grep -qx "TASK1_NNUNET_RESULTS_STAMP_NAME=${FDG_STAMP}"; then
      return 0
    fi
  done
  pgrep -af "nnUNetv2_train" 2>/dev/null | grep -Ev 'pgrep|queue_keeper' | grep -q "${FDG_STAMP}" && return 0
  return 1
}
while [[ ! -f "${FDG_FINAL}" ]]; do
  if _fdg_train_alive; then
    echo "[nnunet-mim] FDG still running $(TZ=Asia/Shanghai date +%H:%M:%S)"
    sleep 90
    continue
  fi
  if FDG_CKPT="$(_resolve_nn_fdg_ckpt)"; then
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
[[ -n "${FDG_CKPT}" && -f "${FDG_CKPT}" ]] || { echo "[error] missing FDG ckpt under ${FDG_FOLD}" >&2; exit 1; }
echo "[nnunet-mim] FDG done; PSMA init=${FDG_CKPT}"

TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}" bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"nnunet_mim\":{\"fdg_pretrain\":{\"status\":\"done\",\"stamp\":\"${FDG_STAMP}\",\"best_ckpt\":\"${FDG_CKPT}\",\"total_epochs\":${FDG_EP},\"note\":\"PET+CT MIM → FDG done · ${FDG_EP}ep\"},\"psma_fs50_f258\":{\"status\":\"running\"}}},\"updated_note\":\"nnUNet MIM FDG done → PSMA\"}" || true

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
export TASK1_BOARD_METHOD=nnunet_mim
unset TASK1_PRETRAINED_WEIGHTS || true
unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

bash "${CTRL}/ICLR2026/run/run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh"
echo "[nnunet-mim] ALL DONE"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD_JSON}" || true
