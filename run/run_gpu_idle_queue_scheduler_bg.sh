#!/usr/bin/env bash
# When any GPU stays idle (low VRAM) for TASK1_GPU_IDLE_SEC (default 60s = 1min),
# launch the next pending task from the aligned board queue on that GPU.
#
#   bash ICLR2026/run/run_gpu_idle_queue_scheduler_bg.sh
#   TASK1_GPU_IDLE_SEC=60 TASK1_GPU_IDLE_MEM_MIB=2048 bash ICLR2026/run/run_gpu_idle_queue_scheduler_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
LOG="${TASK1_GPU_IDLE_LOG:-${VIS}/nohup_gpu_idle_queue_scheduler.log}"
PID_FILE="${VIS}/gpu_idle_queue_scheduler.pid"

export TASK1_GPU_IDLE_SEC="${TASK1_GPU_IDLE_SEC:-60}"
export TASK1_GPU_IDLE_MEM_MIB="${TASK1_GPU_IDLE_MEM_MIB:-2048}"
export TASK1_GPU_IDLE_POLL_SEC="${TASK1_GPU_IDLE_POLL_SEC:-10}"
export TASK1_GPU_IDLE_GPUS="${TASK1_GPU_IDLE_GPUS:-0,1,3}"

if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[gpu-idle-queue] already running pid=${old} idle_sec=${TASK1_GPU_IDLE_SEC} — kill to restart with new settings"
    exit 0
  fi
fi

nohup python3 "${ROOT}/ICLR2026/scripts/gpu_idle_queue_scheduler.py" >>"${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "[gpu-idle-queue] started pid=$(cat "${PID_FILE}") idle=${TASK1_GPU_IDLE_SEC}s mem<${TASK1_GPU_IDLE_MEM_MIB}MiB log=${LOG}"
