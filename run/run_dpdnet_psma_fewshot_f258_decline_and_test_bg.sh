#!/usr/bin/env bash
# DpDNet PSMA f258: resume from base ep until val-Dice decline, then TEST20.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
MON="${ROOT}/ICLR2026/scripts/monitor_val_dice_decline_stop.py"

FEWSHOT_N="${TASK1_FEWSHOT_N:?need TASK1_FEWSHOT_N}"
BOARD_STAGE="${TASK1_PSMA_BOARD_STAGE:-psma_fs${FEWSHOT_N}_f258}"
DPD_PARENT="${DPD_PARENT:?need DPD_PARENT}"
BASE_EP="${TASK1_RESUME_BASE_EP:-100}"
MAX_EP="${TASK1_RESUME_MAX_EPOCHS:-300}"
VAL_EVERY="${TASK1_FS50_VAL_EVERY_N_EPOCHS:-20}"
FOLD_GPUS="${TASK1_FOLD_GPUS:-2:0,5:1,8:3}"
DPD_FOLDS="${DPD_RESUME_FOLDS:-2,5,8}"

PIPE_LOG="${VIS}/nohup_dpdnet_fs${FEWSHOT_N}_decline_${DPD_PARENT}.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

declare -A GPU_OF
IFS=',' read -r -a _pairs <<< "${FOLD_GPUS}"
for p in "${_pairs[@]}"; do
  GPU_OF["${p%%:*}"]="${p##*:}"
done

_start_monitor() {
  local fold="$1"
  local log="${VIS}/nohup_decline_mon_dpdnet_fs${FEWSHOT_N}_f${fold}.log"
  nohup python3 "${MON}" \
    --method dpdnet \
    --parent-stamp "${DPD_PARENT}" \
    --fold "${fold}" \
    --base-ep "${BASE_EP}" \
    --val-every "${VAL_EVERY}" \
    >"${log}" 2>&1 &
  echo $! > "${VIS}/decline_mon_dpdnet_fs${FEWSHOT_N}_f${fold}.pid"
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
    echo "[dpd-decline] finished=${done}/${#stamps[@]}"
    [[ "${done}" -ge "${#stamps[@]}" ]] && break
    sleep 60
  done
}

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"dpdnet\":{\"${BOARD_STAGE}\":{\"status\":\"running\",\"stamp\":\"${DPD_PARENT}\",\"phase\":\"decline\",\"note\":\"resume@${BASE_EP} until val Dice decline\"}}},\"updated_note\":\"DpDNet fs${FEWSHOT_N} decline resume\"}" || true

IFS=',' read -r -a _dpf <<< "${DPD_FOLDS}"
dpd_stamps=()
for fold in "${_dpf[@]}"; do
  gpu="${GPU_OF[${fold}]:-}"
  [[ -n "${gpu}" ]] || { echo "[error] no GPU for dpd fold ${fold}" >&2; exit 1; }
  st="${DPD_PARENT}_f${fold}"
  dpd_stamps+=("${st}")
  rm -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${st}.txt"
  FOLD_ID="${fold}" GPU_ID="${gpu}" PARENT_STAMP="${DPD_PARENT}" \
    TASK1_DPDNET_NUM_EPOCHS="${MAX_EP}" \
    TASK1_DPDNET_TRAIN_ITERS=25 \
    TASK1_DPDNET_VAL_ITERS=25 \
    TASK1_DPDNET_VAL_EVERY="${VAL_EVERY}" \
    TASK1_VAL_EVERY_N_EPOCHS="${VAL_EVERY}" \
    TASK1_BEST_BY=ema_fg_dice \
    TASK1_DPDNET_SKIP_PREPARE=1 \
    bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_fewshot50_onefold_bg.sh"
  sleep 8
  _start_monitor "${fold}"
done
_wait_fold_stamps "${dpd_stamps[@]}"
for st in "${dpd_stamps[@]}"; do
  TASK1_NNUNET_RESULTS_STAMP_NAME="${st}" bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
done

export PARENT_STAMP="${DPD_PARENT}"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${DPD_PARENT}"
export TASK1_FOLDS="${DPD_FOLDS}"
export TASK1_FOLD_GPUS="${FOLD_GPUS}"
export TASK1_TEST_SKIP_DONE=0
bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_test20_f258_parallel.sh"

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"dpdnet\":{\"${BOARD_STAGE}\":{\"status\":\"done\",\"stamp\":\"${DPD_PARENT}\",\"test_invalidated\":false,\"phase\":null,\"note\":\"TEST20 DONE · 3/3\"}}},\"updated_note\":\"DpDNet fs${FEWSHOT_N} TEST20 done\"}" || true

echo "[dpd-decline] ALL DONE fs${FEWSHOT_N} ${DPD_PARENT}"
