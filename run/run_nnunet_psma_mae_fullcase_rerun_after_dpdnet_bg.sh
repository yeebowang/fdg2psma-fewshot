#!/usr/bin/env bash
# Queue: wait until DpDNet PSMA train+TEST20 fully finish, then rerun nnUNet
# PSMA fs50 f258 + TEST20 only (keep FDG). MAE-style:
#   full-case tr=floor(n/bs) / val=ceil(n/bs) every 20ep · 100ep · gbs=2
#   best = max ema_fg_dice → TEST20 uses checkpoint_best (skip nan-best)
#
#   bash ICLR2026/run/run_nnunet_psma_mae_fullcase_rerun_after_dpdnet_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
ICLR_VIS="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
mkdir -p "${ICLR_VIS}"

PIPE_LOG="${ICLR_VIS}/nohup_nnunet_psma_mae_fullcase_rerun_after_dpdnet.log"
if [[ -t 1 ]]; then
  exec > >(tee -a "${PIPE_LOG}") 2>&1
fi

DPD_PARENT="${TASK1_DPDNET_PSMA_PARENT:-}"
if [[ -z "${DPD_PARENT}" && -f "${ICLR_VIS}/dpdnet_psma_fs50_f258_LAST_STAMP.txt" ]]; then
  DPD_PARENT="$(tr -d '[:space:]' < "${ICLR_VIS}/dpdnet_psma_fs50_f258_LAST_STAMP.txt")"
fi
DPD_PARENT="${DPD_PARENT:-20260817_210749_iclr2026_dpdnet_psma_fs50_f258_1gpu_bs2_tr25_val25e20_100ep_gpu013}"

FDG_STAMP="${TASK1_UDA_FDG_STAMP:-}"
if [[ -z "${FDG_STAMP}" && -f "${ICLR_VIS}/baseline1_fdg_LAST_STAMP.txt" ]]; then
  FDG_STAMP="$(tr -d '[:space:]' < "${ICLR_VIS}/baseline1_fdg_LAST_STAMP.txt")"
fi
FDG_STAMP="${FDG_STAMP:-20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep}"
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
PSMA_BS="${TASK1_NNUNET_PSMA_BATCH:-2}"
VAL_EVERY="${TASK1_NNUNET_VAL_EVERY:-20}"
PSMA_SPLITS="${CTRL}/ICLR2026/data/splits_mae_psma_fewshot50_9fold/fold2_nnunet.json"
read -r PSMA_TR _PSMA_VAL_COMPUTED PSMA_NTR PSMA_NVAL < <(
  python3 - <<PY
import json, math
from pathlib import Path
d = json.loads(Path("${PSMA_SPLITS}").read_text())[0]
bs = int("${PSMA_BS}")
print(max(1, len(d["train"]) // bs), max(1, math.ceil(len(d["val"]) / bs)), len(d["train"]), len(d["val"]))
PY
)
PSMA_VAL="${TASK1_NNUNET_PSMA_VAL_ITERS:-25}"

DPD_DS="Dataset240_DpDNet_PSMA_2ch"
DPD_TF="STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres"

echo "[nnunet-psma-rerun] wait DpDNet PARENT=${DPD_PARENT} train+TEST"
echo "[nnunet-psma-rerun] then nnUNet PSMA tr${PSMA_TR}/val${PSMA_VAL}e${VAL_EVERY} ${PSMA_EP}ep gbs=${PSMA_BS} n=${PSMA_NTR}/${PSMA_NVAL} best=ema_fg_dice init=$(basename "${FDG_CKPT}")"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"nnunet\":{\"psma_fs50_f258\":{\"status\":\"queued\",\"stamp\":\"\",\"phase\":null,\"epoch\":null,\"eta\":null,\"eta_sec\":null,\"fold_dice\":{},\"mean\":null,\"fold_ckpt_ep\":{},\"test_live\":null,\"test_invalidated\":true,\"total_epochs\":${PSMA_EP},\"train_iters\":${PSMA_TR},\"val_iters\":${PSMA_VAL},\"gbs\":${PSMA_BS},\"bs\":${PSMA_BS},\"online_val\":\"VAL${PSMA_VAL} every${VAL_EVERY}\",\"metric\":\"TEST20 Dice; best=max ema_fg_dice\",\"note\":\"queued · MAE fullcase after DpDNet\",\"ckpt\":\"\"}}},\"queue\":[\"dpdnet.psma+test\",\"nnunet.psma MAE-fullcase rerun\",\"nnunet.test20\"],\"updated_note\":\"nnUNet PSMA queued · MAE fullcase (wait DpDNet done)\"}" || true

_dpd_finals_ok() {
  local f fd
  for f in 2 5 8; do
    fd="${WORK}/nnUNet_results/${DPD_PARENT}_f${f}/${DPD_DS}/${DPD_TF}/fold_${f}/checkpoint_final.pth"
    [[ -f "${fd}" ]] || return 1
  done
  return 0
}

_dpd_test_ok() {
  python3 - <<PY
import json
from pathlib import Path
parent = "${DPD_PARENT}"
cands = [
    Path("${ICLR_VIS}") / f"aggregate_dpdnet_psma_test20_f258_{parent}.json",
    Path("${WORK}/nnUNet_results") / parent / "aggregate_test20_dice_f258.json",
]
for p in cands:
    if not p.is_file():
        continue
    try:
        ad = json.loads(p.read_text())
    except Exception:
        continue
    folds = ad.get("folds") or {}
    n = 0
    for f in ("2", "5", "8"):
        fv = folds.get(f) or folds.get(int(f))
        if isinstance(fv, dict):
            d = fv.get("test_dice")
            if isinstance(d, (int, float)) and d == d:
                n += 1
    if n >= 3:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

_dpd_gpu_busy() {
  docker ps --format '{{.Names}}' 2>/dev/null | grep -Eq 'dpdnet_psma|nnunet.*test20|uda_predict' && return 0
  return 1
}

if [[ "${TASK1_NNUNET_SKIP_DPD_WAIT:-0}" == "1" ]]; then
  echo "[nnunet-psma-rerun] skip DpDNet wait (TASK1_NNUNET_SKIP_DPD_WAIT=1)"
else
  while true; do
    trn=0
    _dpd_finals_ok && trn=1
    tes=0
    _dpd_test_ok && tes=1
    echo "[nnunet-psma-rerun] wait dpdnet train=${trn} test=${tes} $(date '+%H:%M:%S')"
    if [[ "${trn}" -eq 1 && "${tes}" -eq 1 ]]; then
      if _dpd_gpu_busy; then
        echo "[nnunet-psma-rerun] TEST files ready but GPU still busy; wait"
        sleep 30
        continue
      fi
      break
    fi
    sleep 60
  done
fi

echo "[nnunet-psma-rerun] DpDNet fully done; disarm then nnUNet PSMA"
for f in 2 5 8; do
  TASK1_NNUNET_RESULTS_STAMP_NAME="${DPD_PARENT}_f${f}" \
    bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
done
TASK1_NNUNET_RESULTS_STAMP_NAME="${DPD_PARENT}" \
  bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
sleep 5

PARENT="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_nnunet_psma_fs50_f258_1gpu_bs${PSMA_BS}_tr${PSMA_TR}_val${PSMA_VAL}e${VAL_EVERY}_${PSMA_EP}ep_gpu013"
echo "${PARENT}" > "${ICLR_VIS}/nnunet_psma_fs50_f258_LAST_STAMP.txt"
echo "[nnunet-psma-rerun] PARENT=${PARENT}"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"nnunet\":{\"fdg_pretrain\":{\"status\":\"done\",\"stamp\":\"${FDG_STAMP}\",\"best_ckpt\":\"${FDG_CKPT}\"},\"psma_fs50_f258\":{\"status\":\"running\",\"stamp\":\"${PARENT}\",\"phase\":null,\"epoch\":0,\"total_epochs\":${PSMA_EP},\"train_iters\":${PSMA_TR},\"val_iters\":${PSMA_VAL},\"gbs\":${PSMA_BS},\"bs\":${PSMA_BS},\"test_invalidated\":true,\"fold_dice\":{},\"mean\":null,\"fold_ckpt_ep\":{},\"test_live\":null,\"ckpt\":\"checkpoint_best.pth\",\"metric\":\"TEST20 Dice; best=max ema_fg_dice\",\"note\":\"MAE fullcase rerun · tr${PSMA_TR}/val${PSMA_VAL}e${VAL_EVERY} · best=ema_fg_dice\",\"online_val\":\"VAL${PSMA_VAL} every${VAL_EVERY}\"}}},\"queue\":[\"nnunet.psma MAE-fullcase rerun\",\"nnunet.test20\"],\"updated_note\":\"nnUNet PSMA MAE-fullcase rerun ${PARENT}\"}" || true

# keep DISARMED during PSMA train (guard is still started by onefold)
(
  while true; do
    done_n=0
    for f in 2 5 8; do
      fd="${WORK}/nnUNet_results/${PARENT}_f${f}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
      if [[ -f "${fd}" ]]; then
        done_n=$((done_n + 1))
      else
        TASK1_NNUNET_RESULTS_STAMP_NAME="${PARENT}_f${f}" \
          bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" >/dev/null 2>&1 || true
      fi
    done
    [[ "${done_n}" -ge 3 ]] && break
    sleep 45
  done
) &
DISARM_PID=$!

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
export TASK1_LOSS_PLOT_VAL_EMA=1
export TASK1_FOLDS=2,5,8
export TASK1_FOLD_GPUS=2:0,5:1,8:3
export TASK1_TEST_SKIP_DONE=0
export TASK1_NNUNET_RESULTS_STAMP_NAME="${PARENT}"
export TASK1_CONTINUE_TRAINING=0
export TASK1_CONTINUE_PICK_NEWER=0
export TASK1_CONTINUE_FROM_LATEST=0
export TASK1_CONTINUE_FROM_BEST=0
export TASK1_GUARD_REQUIRE_ARM=1
unset TASK1_PRETRAINED_WEIGHTS || true

bash "${CTRL}/ICLR2026/run/run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh"

kill "${DISARM_PID}" 2>/dev/null || true
wait "${DISARM_PID}" 2>/dev/null || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"queue\":[],\"updated_note\":\"nnUNet PSMA MAE-fullcase rerun TEST20 done · ${PARENT}\"}" || true
echo "[nnunet-psma-rerun] ALL DONE ${PARENT}"
