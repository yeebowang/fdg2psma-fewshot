#!/usr/bin/env bash
# Launch rescore Dice/FP/FN queue in background (after fc70 / fdg-eval).
#
#   bash ICLR2026/run/run_rescore_dice_fp_fn_after_queues_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
PID_FILE="${VIS}/rescore_metrics_queue.pid"
LOG="${VIS}/nohup_rescore_dice_fp_fn_queue.log"
WORKER="${ROOT}/ICLR2026/run/run_rescore_dice_fp_fn_queue_worker.sh"

if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[rescore-queue] already running pid=${old} log=${LOG}"
    exit 0
  fi
fi

chmod +x "${WORKER}"
nohup bash "${WORKER}" >>"${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "[rescore-queue] started pid=$(cat "${PID_FILE}") log=${LOG}"
echo "[rescore-queue] policy: CPU rescore NOW (no GPU wait); GPU MAE/MONAI only after fc70 frees GPUs"
