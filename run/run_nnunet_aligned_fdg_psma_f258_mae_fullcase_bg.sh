#!/usr/bin/env bash
# nnUNet 全流程对齐 PET/CT MAE：
#   FDG: full-case tr/val · 100ep · gbs=6 · val every20 · best=max ema_fg_dice（≈MAE SWI Dice）
#   PSMA fs50 f258: full-case tr/val · 100ep · gbs=2 · val every20 · best=ema_fg_dice → TEST20
#   PSMA init = FDG checkpoint_best（非 final）
#
#   bash ICLR2026/run/run_nnunet_aligned_fdg_psma_f258_mae_fullcase_bg.sh
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

FDG_EP="${TASK1_NNUNET_FDG_EPOCHS:-100}"
PSMA_EP="${TASK1_NNUNET_PSMA_EPOCHS:-${TASK1_NUM_EPOCHS:-100}}"
FDG_BS="${TASK1_NNUNET_FDG_BATCH:-6}"
# Do not inherit FDG TASK1_FIXED_BATCH_3D_FULLRES for PSMA
PSMA_BS="${TASK1_NNUNET_PSMA_BATCH:-2}"
VAL_EVERY="${TASK1_NNUNET_VAL_EVERY:-20}"
FDG_SPLITS="${TASK1_SPLITS_FINAL_JSON:-${CTRL}/ICLR2026/data/splits_baseline1_fdg_nnunet.json}"
PSMA_SPLITS="${CTRL}/ICLR2026/data/splits_mae_psma_fewshot50_9fold/fold2_nnunet.json"

# full-case：train=floor(n/bs)（对齐 MAE drop_last）；val=ceil(n/bs)（扫完所有 case）
read -r FDG_TR FDG_VAL FDG_NTR FDG_NVAL < <(
  python3 - <<PY
import json, math
from pathlib import Path
d = json.loads(Path("${FDG_SPLITS}").read_text())[0]
ntr, nva = len(d["train"]), len(d["val"])
bs = int("${FDG_BS}")
print(max(1, ntr // bs), max(1, math.ceil(nva / bs)), ntr, nva)
PY
)
read -r PSMA_TR _PSMA_VAL_COMPUTED PSMA_NTR PSMA_NVAL < <(
  python3 - <<PY
import json, math
from pathlib import Path
d = json.loads(Path("${PSMA_SPLITS}").read_text())[0]
ntr, nva = len(d["train"]), len(d["val"])
bs = int("${PSMA_BS}")
print(max(1, ntr // bs), max(1, math.ceil(nva / bs)), ntr, nva)
PY
)
# aligned DpDNet/nnUNet PSMA fewshot: val25 every20 (not ceil(n/bs)=30)
PSMA_VAL="${TASK1_NNUNET_PSMA_VAL_ITERS:-25}"

PIPE_LOG="${ICLR_VIS}/nohup_nnunet_aligned_mae_fullcase_fdg${FDG_EP}_psma_f258.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[nnunet-mae-fullcase] FDG ${FDG_EP}ep tr${FDG_TR}/val${FDG_VAL}e${VAL_EVERY} (n=${FDG_NTR}/${FDG_NVAL} gbs=${FDG_BS}) best=ema_fg_dice"
echo "[nnunet-mae-fullcase] PSMA ${PSMA_EP}ep tr${PSMA_TR}/val${PSMA_VAL}e${VAL_EVERY} (n=${PSMA_NTR}/${PSMA_NVAL} gbs=${PSMA_BS}) → TEST20"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"protocol\":\"FDG→PSMA fs50 f258; align MAE: full-case · ${FDG_EP}ep · best=val Dice · TEST20\",\"methods\":{\"nnunet\":{\"fdg_pretrain\":{\"status\":\"running\",\"total_epochs\":${FDG_EP},\"train_iters\":${FDG_TR},\"val_iters\":${FDG_VAL},\"gbs\":${FDG_BS},\"bs\":${FDG_BS},\"bs_note\":\"gbs\",\"note\":\"MAE-align · fullcase tr${FDG_TR}/val${FDG_VAL}e${VAL_EVERY} · best=ema_fg_dice\",\"train_sec\":null,\"train_time\":null,\"best_ep\":null,\"epoch\":null,\"stamp\":\"\",\"eta\":null,\"eta_sec\":null},\"psma_fs50_f258\":{\"status\":\"queued\",\"stamp\":\"\",\"total_epochs\":${PSMA_EP},\"train_iters\":${PSMA_TR},\"val_iters\":${PSMA_VAL},\"online_val\":\"VAL${PSMA_VAL} every${VAL_EVERY}\",\"gbs\":${PSMA_BS},\"bs\":${PSMA_BS},\"bs_note\":\"gbs\",\"metric\":\"TEST20 Dice; best=max ema_fg_dice\",\"note\":\"queued · MAE fullcase\",\"epoch\":null,\"train_sec\":null,\"train_time\":null,\"fold_dice\":{},\"mean\":null,\"fold_ckpt_ep\":{},\"test_invalidated\":true}}},\"queue\":[\"nnunet.fdg fullcase ${FDG_EP}ep\",\"nnunet.psma f258\",\"nnunet.test20\",\"dpdnet.fullcase\"],\"updated_note\":\"nnUNet MAE-fullcase FDG tr${FDG_TR}/val${FDG_VAL}e${VAL_EVERY} ${FDG_EP}ep\"}" || true

# --- FDG ---
export TASK1_NUM_EPOCHS="${FDG_EP}"
export TASK1_TRAIN_ITERS_PER_EPOCH="${FDG_TR}"
export TASK1_VAL_ITERS_PER_EPOCH="${FDG_VAL}"
export TASK1_VAL_EVERY_N_EPOCHS="${VAL_EVERY}"
export TASK1_FIXED_BATCH_3D_FULLRES="${FDG_BS}"
export TASK1_BEST_BY=ema_fg_dice
export TASK1_VAL_LOSS_ONLY=0
export TASK1_LOSS_PLOT_VAL_EMA=1
export TASK1_PSMA_VAL_ENABLE=0
export TASK1_VAL_ITERS_LATE_FROM_EPOCH=999999
export TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH=0
export TASK1_LOSS_PLOT_VAL_FROM_EPOCH=1
export TASK1_SPLITS_FINAL_JSON="${FDG_SPLITS}"
# Fresh FDG launch must not inherit CONTINUE=1 from parent shell / guard EXTRA
export TASK1_CONTINUE_TRAINING=0
export TASK1_CONTINUE_PICK_NEWER=0
export TASK1_CONTINUE_FROM_LATEST=0
export TASK1_CONTINUE_FROM_BEST=0
unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

bash "${CTRL}/ICLR2026/run/run_baseline1_fdg_2ch_fullres_3000ep_bg.sh"

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

_resolve_nn_fdg_ckpt() {
  local f
  for f in checkpoint_best.pth checkpoint_final.pth checkpoint_latest.pth; do
    if [[ -f "${FDG_FOLD}/${f}" ]]; then
      echo "${FDG_FOLD}/${f}"
      return 0
    fi
  done
  return 1
}

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"nnunet\":{\"fdg_pretrain\":{\"status\":\"running\",\"stamp\":\"${FDG_STAMP}\",\"total_epochs\":${FDG_EP},\"train_iters\":${FDG_TR},\"val_iters\":${FDG_VAL},\"gbs\":${FDG_BS},\"bs\":${FDG_BS},\"note\":\"MAE-align · fullcase · best=ema_fg_dice · e${VAL_EVERY}\"}}},\"updated_note\":\"nnUNet FDG running · ${FDG_STAMP}\"}" || true

echo "[nnunet-mae-fullcase] wait FDG ${FDG_STAMP} (prefer checkpoint_best after final)…"
while [[ ! -f "${FDG_FINAL}" ]]; do
  if ! docker ps --format '{{.Names}}' | grep -qi baseline1; then
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
[[ -n "${FDG_CKPT}" && -f "${FDG_CKPT}" ]] || { echo "[error] missing FDG best/final/latest under ${FDG_FOLD}" >&2; exit 1; }
echo "[nnunet-mae-fullcase] FDG done; PSMA init=${FDG_CKPT}"

TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}" bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"nnunet\":{\"fdg_pretrain\":{\"status\":\"done\",\"stamp\":\"${FDG_STAMP}\",\"best_ckpt\":\"${FDG_CKPT}\",\"total_epochs\":${FDG_EP},\"note\":\"MAE-align FDG done · init=$(basename "${FDG_CKPT}")\"},\"psma_fs50_f258\":{\"status\":\"running\",\"note\":\"MAE-align PSMA after FDG (init=best)\"}}},\"updated_note\":\"nnUNet FDG ${FDG_EP}ep done -> PSMA\",\"queue\":[\"nnunet.psma f258\",\"nnunet.test20\",\"dpdnet.fullcase\"]}" || true

# --- PSMA + TEST ---
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

bash "${CTRL}/ICLR2026/run/run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh"

echo "[nnunet-mae-fullcase] nnUNet ALL DONE"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json '{"queue":["dpdnet.fullcase"],"updated_note":"nnUNet MAE-fullcase FDG+PSMA+TEST done"}' || true
