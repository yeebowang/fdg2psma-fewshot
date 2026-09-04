#!/usr/bin/env bash
# Queue PSMA fs0 + FDG TEST eval after fc70% pipeline (or when prior queues done).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
POLL_SEC="${TASK1_CHAIN_POLL_SEC:-60}"
PID_FILE="${VIS}/fdg_eval_queue.pid"
LOG="${VIS}/nohup_fdg_eval_queue.log"

echo $$ > "${PID_FILE}"
exec > >(tee -a "${LOG}") 2>&1

echo "[fdg-eval-queue] $(date '+%F %T') start pid=$$"

_wait_prior() {
  while pgrep -f "run_aligned_psma_fc70_pipeline_bg.sh" >/dev/null 2>&1 \
     || pgrep -f "run_nnunet_psma_fc70" >/dev/null 2>&1 \
     || pgrep -f "run_mae_psma_fc70" >/dev/null 2>&1 \
     || pgrep -f "run_monai_psma_fc70" >/dev/null 2>&1 \
     || pgrep -f "run_dpdnet_psma_fc70" >/dev/null 2>&1; do
    echo "[fdg-eval-queue] waiting fc70 pipeline… $(TZ=Asia/Shanghai date +%H:%M:%S)"
    sleep "${POLL_SEC}"
  done
  if [[ -f "${VIS}/psma_fc70_queue.pid" ]]; then
    pid=$(tr -d '[:space:]' < "${VIS}/psma_fc70_queue.pid" 2>/dev/null || true)
    while [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; do
      echo "[fdg-eval-queue] waiting fc70 queue pid=${pid}…"
      sleep "${POLL_SEC}"
    done
  fi
  while pgrep -f "run_aligned_psma_fs10_fs5_pipeline_bg.sh" >/dev/null 2>&1 \
     || pgrep -f "run_nnunet_psma_fs10_fs5_rerun_fdg169" >/dev/null 2>&1; do
    echo "[fdg-eval-queue] waiting fs10/fs5 prior…"
    sleep "${POLL_SEC}"
  done
}

_wait_eval() {
  local pat="$1"
  if pgrep -f "${pat}" >/dev/null 2>&1; then
    echo "[fdg-eval-queue] ${pat} already running — wait"
    while pgrep -f "${pat}" >/dev/null 2>&1; do
      sleep "${POLL_SEC}"
    done
  fi
}

_wait_prior

_wait_eval "run_eval_fdg_shared_test20_bg.sh"
echo "[fdg-eval-queue] launching PSMA fs0 eval"
bash "${ROOT}/ICLR2026/run/run_eval_fdg_shared_test20_bg.sh"

_wait_eval "run_eval_fdg_test20_bg.sh"
echo "[fdg-eval-queue] launching FDG TEST eval"
bash "${ROOT}/ICLR2026/run/run_eval_fdg_test20_bg.sh"

echo "[fdg-eval-queue] ALL DONE"
