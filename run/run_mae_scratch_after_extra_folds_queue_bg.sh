#!/usr/bin/env bash
# Queue tail: after fs50/10/5 extra-folds → 9fold, run PET/CT MAE scratch (9-fold).
#
#   bash ICLR2026/run/run_mae_scratch_after_extra_folds_queue_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
POLL_SEC="${TASK1_CHAIN_POLL_SEC:-60}"
DONE_EXTRA="${VIS}/TASK1_PSMA_EXTRA_FOLDS_9FOLD_DONE.txt"
DONE_SELF="${VIS}/TASK1_MAE_SCRATCH_9FOLD_DONE.txt"
PID_FILE="${VIS}/mae_scratch_9fold_queue.pid"
LOG="${VIS}/nohup_mae_scratch_9fold_queue.log"

mkdir -p "${VIS}"
if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" && "${old}" != "$$" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[mae-scratch-queue] already running pid=${old}"
    exit 0
  fi
fi
echo $$ > "${PID_FILE}"
exec > >(tee -a "${LOG}") 2>&1

echo "[mae-scratch-queue] $(date '+%F %T') start pid=$$"

if [[ -f "${DONE_SELF}" ]] && grep -q 'status=ok' "${DONE_SELF}"; then
  echo "[mae-scratch-queue] already done (marker)"
  exit 0
fi

_pgrep_real() {
  local pat="$1"
  pgrep -af "$pat" 2>/dev/null | grep -Ev 'queue_keeper|pgrep|run_mae_scratch_after_extra_folds' | grep -q .
}

echo "[mae-scratch-queue] waiting extra folds → 9fold DONE"
while true; do
  if [[ -f "${DONE_EXTRA}" ]] && grep -q 'status=ok' "${DONE_EXTRA}"; then
    if ! _pgrep_real 'run_aligned_psma_extra_fold_onegpu.sh' \
      && ! _pgrep_real 'run_aligned_psma_extra_folds_9fold_pipeline'; then
      echo "[mae-scratch-queue] extra folds done"
      break
    fi
  fi
  echo "[mae-scratch-queue] waiting extra folds… $(TZ=Asia/Shanghai date +%H:%M:%S)"
  sleep "${POLL_SEC}"
done

echo "[mae-scratch-queue] launch MAE scratch 9fold pipeline"
bash "${ROOT}/ICLR2026/run/run_aligned_mae_scratch_9fold_pipeline_bg.sh"
echo "[mae-scratch-queue] ALL DONE $(date '+%F %T')"
