#!/usr/bin/env bash
# Queue: after fc70 + FDG TEST eval, run fs50/fs10/fs5 extra folds → 9-fold pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
POLL_SEC="${TASK1_CHAIN_POLL_SEC:-60}"
DONE_MARK="${VIS}/TASK1_PSMA_EXTRA_FOLDS_9FOLD_DONE.txt"
PID_FILE="${VIS}/extra_folds_9fold_queue.pid"
LOG="${VIS}/nohup_extra_folds_9fold_queue.log"

echo $$ > "${PID_FILE}"
exec > >(tee -a "${LOG}") 2>&1

_pgrep_real() {
  local pat="$1"
  pgrep -af "$pat" 2>/dev/null | grep -Ev 'queue_keeper|run_psma_extra_folds_9fold_after_eval|run_aligned_psma_extra_folds_9fold' | grep -q .
}

_wait_prior() {
  if [[ -f "${DONE_MARK}" ]] && grep -q 'status=ok' "${DONE_MARK}"; then
    echo "[extra-9fold-queue] already done (marker)"
    return 0
  fi
  # Extra folds are launched per idle GPU by gpu-idle scheduler (do not wait for fc70).
  # Only fall back to the bulk 3-GPU pipeline if idle scheduler is dead.
  if pgrep -af 'gpu_idle_queue_scheduler.py' 2>/dev/null | grep -Ev 'queue_keeper|pgrep' | grep -q .; then
    echo "[extra-9fold-queue] gpu-idle scheduler alive — extra folds go to idle GPUs; not launching bulk pipeline"
    while [[ ! -f "${DONE_MARK}" ]] || ! grep -q 'status=ok' "${DONE_MARK}"; do
      if ! pgrep -af 'gpu_idle_queue_scheduler.py' 2>/dev/null | grep -Ev 'queue_keeper|pgrep' | grep -q .; then
        echo "[extra-9fold-queue] idle scheduler gone — fall back to bulk pipeline"
        break
      fi
      echo "[extra-9fold-queue] waiting gpu-idle extra folds… $(TZ=Asia/Shanghai date +%H:%M:%S)"
      sleep "${POLL_SEC}"
    done
    if [[ -f "${DONE_MARK}" ]] && grep -q 'status=ok' "${DONE_MARK}"; then
      return 0
    fi
  fi
  while _pgrep_real 'run_aligned_psma_fc70_pipeline_bg.sh' \
     || _pgrep_real 'run_nnunet_psma_fc70' \
     || _pgrep_real 'run_mae_psma_fc70' \
     || _pgrep_real 'run_monai_psma_fc70' \
     || _pgrep_real 'run_dpdnet_psma_fc70' \
     || _pgrep_real 'run_seganypet_psma_fc70'; do
    echo "[extra-9fold-queue] waiting fc70… $(TZ=Asia/Shanghai date +%H:%M:%S)"
    sleep "${POLL_SEC}"
  done
  if [[ -f "${VIS}/psma_fc70_queue.pid" ]]; then
    pid=$(tr -d '[:space:]' < "${VIS}/psma_fc70_queue.pid" 2>/dev/null || true)
    while [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; do
      echo "[extra-9fold-queue] waiting fc70 queue pid=${pid}…"
      sleep "${POLL_SEC}"
    done
  fi
  while _pgrep_real 'run_eval_fdg_shared_test20_bg.sh' \
     || _pgrep_real 'run_eval_fdg_test20_bg.sh' \
     || _pgrep_real 'run_fdg_eval_after_fc70_queue_bg.sh' \
     || _pgrep_real 'run_fdg20_test_after_fc70'; do
    echo "[extra-9fold-queue] waiting FDG TEST / fs0 eval… $(TZ=Asia/Shanghai date +%H:%M:%S)"
    sleep "${POLL_SEC}"
  done
  if [[ -f "${VIS}/fdg_eval_queue.pid" ]]; then
    pid=$(tr -d '[:space:]' < "${VIS}/fdg_eval_queue.pid" 2>/dev/null || true)
    while [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; do
      echo "[extra-9fold-queue] waiting fdg eval queue pid=${pid}…"
      sleep "${POLL_SEC}"
    done
  fi
}

echo "[extra-9fold-queue] $(date '+%F %T') start pid=$$"
_wait_prior
bash "${ROOT}/ICLR2026/run/run_aligned_psma_extra_folds_9fold_pipeline_bg.sh"
echo "[extra-9fold-queue] ALL DONE"
