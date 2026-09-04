#!/usr/bin/env bash
# Resume nnUNet PSMA f258 folds from base ep until val-Dice decline, then TEST20.
# Required: NN_PARENT, TASK1_FEWSHOT_N (10/5/50)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
MON="${ROOT}/ICLR2026/scripts/monitor_val_dice_decline_stop.py"

FEWSHOT_N="${TASK1_FEWSHOT_N:?need TASK1_FEWSHOT_N}"
BOARD_STAGE="${TASK1_PSMA_BOARD_STAGE:-psma_fs${FEWSHOT_N}_f258}"
NN_PARENT="${NN_PARENT:?need NN_PARENT}"
BASE_EP="${TASK1_RESUME_BASE_EP:-100}"
MAX_EP="${TASK1_RESUME_MAX_EPOCHS:-300}"
VAL_EVERY="${TASK1_FS50_VAL_EVERY_N_EPOCHS:-20}"
FOLD_GPUS="${TASK1_FOLD_GPUS:-2:0,5:1,8:3}"
NN_FOLDS="${NN_RESUME_FOLDS:-2,5,8}"
SPLIT_DIR="${TASK1_FEWSHOT_SPLIT_DIR:-${ROOT}/ICLR2026/data/splits_mae_psma_fewshot${FEWSHOT_N}_9fold}"

FDG_STAMP="${TASK1_UDA_FDG_STAMP:-20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep}"
FDG_BEST="${TASK1_UDA_FDG_BEST:-/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth}"
if [[ -z "${FDG_BEST}" ]]; then
  _fold="${WORK}/nnUNet_results/${FDG_STAMP}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0"
  for _c in checkpoint_best.pth checkpoint_final.pth checkpoint_latest.pth; do
    [[ -f "${_fold}/${_c}" ]] && { FDG_BEST="${_fold}/${_c}"; break; }
  done
fi

PIPE_LOG="${VIS}/nohup_nnunet_fs${FEWSHOT_N}_decline_${NN_PARENT}.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

declare -A GPU_OF
IFS=',' read -r -a _pairs <<< "${FOLD_GPUS}"
for p in "${_pairs[@]}"; do
  GPU_OF["${p%%:*}"]="${p##*:}"
done

_start_monitor() {
  local fold="$1"
  local log="${VIS}/nohup_decline_mon_nnunet_fs${FEWSHOT_N}_f${fold}.log"
  nohup python3 "${MON}" \
    --method nnunet \
    --parent-stamp "${NN_PARENT}" \
    --fold "${fold}" \
    --base-ep "${BASE_EP}" \
    --val-every "${VAL_EVERY}" \
    >"${log}" 2>&1 &
  echo $! > "${VIS}/decline_mon_nnunet_fs${FEWSHOT_N}_f${fold}.pid"
  echo "[decline] monitor nnunet fs${FEWSHOT_N} f${fold} pid=$(cat "${VIS}/decline_mon_nnunet_fs${FEWSHOT_N}_f${fold}.pid")"
}

_wait_fold_stamps() {
  local stamps=("$@")
  while true; do
    local done=0
    for st in "${stamps[@]}"; do
      if [[ -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${st}.txt" ]]; then
        done=$((done + 1))
        continue
      fi
      if ! pgrep -af "${st}" >/dev/null 2>&1 \
         && ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qF "${st}"; then
        done=$((done + 1))
      fi
    done
    echo "[decline] finished=${done}/${#stamps[@]}"
    [[ "${done}" -ge "${#stamps[@]}" ]] && break
    sleep 60
  done
}

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"nnunet\":{\"${BOARD_STAGE}\":{\"status\":\"running\",\"stamp\":\"${NN_PARENT}\",\"phase\":\"decline\",\"note\":\"resume@${BASE_EP} until val Dice decline\"}}},\"updated_note\":\"nnUNet fs${FEWSHOT_N} decline resume\"}" || true

echo "[decline] nnUNet PARENT=${NN_PARENT} folds=${NN_FOLDS} fs${FEWSHOT_N}"
IFS=',' read -r -a _nnf <<< "${NN_FOLDS}"
nn_stamps=()
for fold in "${_nnf[@]}"; do
  gpu="${GPU_OF[${fold}]:-}"
  [[ -n "${gpu}" ]] || { echo "[error] no GPU for fold ${fold}" >&2; exit 1; }
  st="${NN_PARENT}_f${fold}"
  nn_stamps+=("${st}")
  rm -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${st}.txt"
  FOLD_ID="${fold}" GPU_ID="${gpu}" PARENT_STAMP="${NN_PARENT}" \
    TASK1_FEWSHOT_N="${FEWSHOT_N}" \
    TASK1_FEWSHOT_SPLIT_DIR="${SPLIT_DIR}" \
    TASK1_UDA_FDG_STAMP="${FDG_STAMP}" \
    TASK1_UDA_FDG_BEST="${FDG_BEST}" \
    TASK1_NUM_EPOCHS="${MAX_EP}" \
    TASK1_LR_SCHEDULE_NUM_EPOCHS="${BASE_EP}" \
    TASK1_TRAIN_ITERS_PER_EPOCH=25 \
    TASK1_VAL_ITERS_PER_EPOCH=25 \
    TASK1_FS50_VAL_ITERS=25 \
    TASK1_VAL_EVERY_N_EPOCHS="${VAL_EVERY}" \
    TASK1_FS50_VAL_EVERY_N_EPOCHS="${VAL_EVERY}" \
    TASK1_FIXED_BATCH_3D_FULLRES=2 \
    TASK1_BEST_BY=ema_fg_dice \
    TASK1_VAL_LOSS_ONLY=0 \
    TASK1_CONTINUE_TRAINING=1 \
    TASK1_CONTINUE_FROM_BEST=1 \
    TASK1_CONTINUE_FROM_LATEST=0 \
    TASK1_CONTINUE_PICK_NEWER=1 \
    bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fewshot50_onefold_bg.sh"
  sleep 8
  _start_monitor "${fold}"
done
_wait_fold_stamps "${nn_stamps[@]}"
for st in "${nn_stamps[@]}"; do
  TASK1_NNUNET_RESULTS_STAMP_NAME="${st}" bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
done

echo "[decline] TEST20 nnUNet fs${FEWSHOT_N} PARENT=${NN_PARENT}"
for fold in "${_nnf[@]}"; do
  TASK1_CRASH_MONITOR_STAGE="nnunet_fs${FEWSHOT_N}_f${fold}_before_eval" \
  TASK1_NNUNET_RESULTS_STAMP_NAME="${NN_PARENT}_f${fold}" \
    bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
done

export PARENT_STAMP="${NN_PARENT}"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${NN_PARENT}"
export TASK1_FOLDS="${NN_FOLDS}"
export TASK1_FOLD_GPUS="${FOLD_GPUS}"
export TASK1_UDA_PRED_PER_GPU="${TASK1_UDA_PRED_PER_GPU:-5}"
export TASK1_TEST_SKIP_DONE=0
bash "${ROOT}/ICLR2026/run/run_nnunet_psma_test20_f258_parallel.sh"

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"nnunet\":{\"${BOARD_STAGE}\":{\"status\":\"done\",\"stamp\":\"${NN_PARENT}\",\"test_invalidated\":false,\"phase\":null,\"note\":\"TEST20 DONE · 3/3\"}}},\"updated_note\":\"nnUNet fs${FEWSHOT_N} TEST20 done\"}" || true

echo "[decline] ALL DONE nnUNet fs${FEWSHOT_N} ${NN_PARENT}"
