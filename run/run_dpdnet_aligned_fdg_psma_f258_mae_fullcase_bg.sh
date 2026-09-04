#!/usr/bin/env bash
# DpDNet 全流程对齐 PET/CT MAE（在 nnUNet MAE-fullcase 之后或单独跑）：
#   FDG: full-case · 100ep · gbs=6 · val every20 · best=ema_fg_dice
#   PSMA: full-case · 100ep · gbs=2 · val every20 · best=ema_fg_dice → TEST20
#   PSMA init = FDG checkpoint_best
#
#   bash ICLR2026/run/run_dpdnet_aligned_fdg_psma_f258_mae_fullcase_bg.sh
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

FDG_EP="${TASK1_DPDNET_NUM_EPOCHS:-100}"
PSMA_EP="${TASK1_DPDNET_PSMA_EPOCHS:-100}"
FDG_GBS="${TASK1_DPDNET_FDG_GBS:-6}"
PSMA_BS="${TASK1_DPDNET_BATCH_SIZE:-2}"
VAL_EVERY="${TASK1_DPDNET_VAL_EVERY:-20}"
# Prefer 3 GPU × bs2 = gbs6 (same cards as nnUNet); override via TASK1_DPDNET_GPUS
export TASK1_DPDNET_GPUS="${TASK1_DPDNET_GPUS:-0,1,3}"
N_GPU="$(awk -F',' '{print NF}' <<<"${TASK1_DPDNET_GPUS}")"
export TASK1_DPDNET_BATCH_SIZE_PER_GPU=$((FDG_GBS / N_GPU))
if (( TASK1_DPDNET_BATCH_SIZE_PER_GPU < 1 )); then
  export TASK1_DPDNET_BATCH_SIZE_PER_GPU=1
fi

FDG_SPLITS="${CTRL}/ICLR2026/data/splits_baseline1_fdg_nnunet.json"
PSMA_SPLITS="${CTRL}/ICLR2026/data/splits_mae_psma_fewshot50_9fold/fold2_nnunet.json"

read -r FDG_TR FDG_VAL FDG_NTR FDG_NVAL < <(
  python3 - <<PY
import json, math
from pathlib import Path
d = json.loads(Path("${FDG_SPLITS}").read_text())[0]
ntr, nva = len(d["train"]), len(d["val"])
bs = int("${FDG_GBS}")
print(max(1, ntr // bs), max(1, math.ceil(nva / bs)), ntr, nva)
PY
)
read -r PSMA_TR _PSMA_VAL_COMPUTED < <(
  python3 - <<PY
import json, math
from pathlib import Path
d = json.loads(Path("${PSMA_SPLITS}").read_text())[0]
ntr, nva = len(d["train"]), len(d["val"])
bs = int("${PSMA_BS}")
print(max(1, ntr // bs), max(1, math.ceil(nva / bs)))
PY
)
# aligned DpDNet PSMA fewshot: val25 every20 (mean≈0.395 config)
PSMA_VAL="${TASK1_DPDNET_PSMA_VAL_ITERS:-25}"

PIPE_LOG="${ICLR_VIS}/nohup_dpdnet_aligned_mae_fullcase_fdg${FDG_EP}_psma_f258.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[dpdnet-mae-fullcase] FDG ${FDG_EP}ep tr${FDG_TR}/val${FDG_VAL}e${VAL_EVERY} gbs=${FDG_GBS} (${N_GPU}gpu×${TASK1_DPDNET_BATCH_SIZE_PER_GPU}) best=ema_fg_dice"
echo "[dpdnet-mae-fullcase] PSMA ${PSMA_EP}ep tr${PSMA_TR}/val${PSMA_VAL}e${VAL_EVERY} gbs=${PSMA_BS}"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"dpdnet\":{\"fdg_pretrain\":{\"status\":\"running\",\"total_epochs\":${FDG_EP},\"train_iters\":${FDG_TR},\"val_iters\":${FDG_VAL},\"gbs\":${FDG_GBS},\"bs\":${TASK1_DPDNET_BATCH_SIZE_PER_GPU},\"bs_note\":\"gbs=${FDG_GBS}\",\"note\":\"MAE-align · fullcase tr${FDG_TR}/val${FDG_VAL}e${VAL_EVERY} · best=ema_fg_dice\",\"stamp\":\"\",\"train_sec\":null,\"train_time\":null,\"best_ep\":null,\"epoch\":null,\"eta\":null,\"eta_sec\":null},\"psma_fs50_f258\":{\"status\":\"queued\",\"stamp\":\"\",\"total_epochs\":${PSMA_EP},\"train_iters\":${PSMA_TR},\"val_iters\":${PSMA_VAL},\"online_val\":\"VAL${PSMA_VAL} every${VAL_EVERY}\",\"gbs\":${PSMA_BS},\"bs\":${PSMA_BS},\"metric\":\"TEST20 Dice; best=max ema_fg_dice\",\"note\":\"queued · MAE fullcase\",\"fold_dice\":{},\"mean\":null,\"fold_ckpt_ep\":{},\"test_invalidated\":true}}},\"queue\":[\"dpdnet.fdg fullcase\",\"dpdnet.psma f258\",\"dpdnet.test20\"],\"updated_note\":\"DpDNet MAE-fullcase FDG starting\"}" || true

export TASK1_DPDNET_NUM_EPOCHS="${FDG_EP}"
export TASK1_DPDNET_TRAIN_ITERS="${FDG_TR}"
export TASK1_DPDNET_VAL_ITERS="${FDG_VAL}"
export TASK1_DPDNET_VAL_EVERY="${VAL_EVERY}"
export TASK1_VAL_EVERY_N_EPOCHS="${VAL_EVERY}"
export TASK1_TRAIN_ITERS_PER_EPOCH="${FDG_TR}"
export TASK1_VAL_ITERS_PER_EPOCH="${FDG_VAL}"
export TASK1_BEST_BY=ema_fg_dice
export TASK1_DPDNET_SKIP_PREPARE="${TASK1_DPDNET_SKIP_PREPARE:-0}"
unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

bash "${CTRL}/ICLR2026/run/run_dpdnet_fdg_3gpu_bs2_bg.sh"

FDG_STAMP="$(tr -d '[:space:]' < "${ICLR_VIS}/dpdnet_fdg_LAST_STAMP.txt")"
[[ -n "${FDG_STAMP}" ]] || { echo "[error] no DpDNet FDG stamp" >&2; exit 1; }

FDG_DS="Dataset239_DpDNet_FDG_2ch"
FDG_TF="STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres"
FDG_FOLD="${WORK}/nnUNet_results/${FDG_STAMP}/${FDG_DS}/${FDG_TF}/fold_0"
FDG_FINAL="${FDG_FOLD}/checkpoint_final.pth"

_resolve_dpd_fdg_ckpt() {
  local f
  for f in checkpoint_best.pth checkpoint_final.pth checkpoint_latest.pth; do
    if [[ -f "${FDG_FOLD}/${f}" ]]; then
      echo "${FDG_FOLD}/${f}"
      return 0
    fi
  done
  return 1
}

echo "[dpdnet-mae-fullcase] wait FDG ${FDG_STAMP}…"
while [[ ! -f "${FDG_FINAL}" ]]; do
  cname="dpdnet_fdg_${FDG_STAMP}"
  if ! docker ps --format '{{.Names}}' | grep -qx "${cname}"; then
    if [[ -f "${FDG_FINAL}" ]] || [[ -f "${FDG_FOLD}/checkpoint_latest.pth" ]] || [[ -f "${FDG_FOLD}/checkpoint_best.pth" ]]; then
      break
    fi
  fi
  sleep 90
done
FDG_CKPT="$(_resolve_dpd_fdg_ckpt)" || true
[[ -n "${FDG_CKPT}" && -f "${FDG_CKPT}" ]] || { echo "[error] missing DpDNet FDG ckpt" >&2; exit 1; }
echo "[dpdnet-mae-fullcase] FDG done; PSMA init=${FDG_CKPT}"

TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}" bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"dpdnet\":{\"fdg_pretrain\":{\"status\":\"done\",\"stamp\":\"${FDG_STAMP}\",\"best_ckpt\":\"${FDG_CKPT}\",\"total_epochs\":${FDG_EP},\"note\":\"MAE-align FDG done · init=$(basename "${FDG_CKPT}")\"},\"psma_fs50_f258\":{\"status\":\"running\",\"note\":\"MAE-align PSMA after FDG\"}}},\"updated_note\":\"DpDNet FDG done -> PSMA\",\"queue\":[\"dpdnet.psma f258\",\"dpdnet.test20\"]}" || true

export TASK1_DPDNET_FDG_STAMP="${FDG_STAMP}"
export TASK1_DPDNET_FDG_BEST="${FDG_CKPT}"
export TASK1_DPDNET_SKIP_PREPARE=1
export TASK1_DPDNET_NUM_EPOCHS="${PSMA_EP}"
export TASK1_DPDNET_TRAIN_ITERS="${PSMA_TR}"
export TASK1_DPDNET_VAL_ITERS="${PSMA_VAL}"
export TASK1_DPDNET_VAL_EVERY="${VAL_EVERY}"
export TASK1_VAL_EVERY_N_EPOCHS="${VAL_EVERY}"
export TASK1_BEST_BY=ema_fg_dice
export TASK1_DPDNET_BATCH_SIZE="${PSMA_BS}"
export TASK1_FOLDS=2,5,8
export TASK1_FOLD_GPUS=2:0,5:1,8:3
unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

bash "${CTRL}/ICLR2026/run/run_dpdnet_psma_fewshot50_f258_1gpu_bs2_100ep_bg.sh"

echo "[dpdnet-mae-fullcase] DpDNet ALL DONE"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json '{"queue":[],"updated_note":"DpDNet MAE-fullcase FDG+PSMA+TEST done"}' || true
