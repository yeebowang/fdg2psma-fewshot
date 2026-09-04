#!/usr/bin/env bash
# Queue fc70% PSMA pipeline after fs10/fs5 + nnUNet rerun queue finishes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
POLL_SEC="${TASK1_CHAIN_POLL_SEC:-60}"
DONE_MARK="${VIS}/TASK1_FS10_FS5_METHODS_PIPELINE_DONE.txt"
QUEUE_LOG="${VIS}/nohup_psma_fc70_queue.log"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

_pgrep_real() {
  local pat="$1"
  pgrep -af "$pat" 2>/dev/null | grep -Ev 'queue_keeper|run_psma_fc70_after_main_queue|run_fdg_eval_after_fc70|run_nnunet_psma_fs10_fs5_rerun_fdg169_after' | grep -q .
}

_wait_prior() {
  if [[ -f "${DONE_MARK}" ]] && grep -q 'status=ok' "${DONE_MARK}"; then
    echo "[fc70-queue] fs10/fs5 methods pipeline done (marker)"
    return 0
  fi
  while _pgrep_real 'run_aligned_psma_fs10_fs5_pipeline_bg.sh' \
     || _pgrep_real 'run_nnunet_psma_fc70'; do
    echo "[fc70-queue] waiting prior pipelines… $(TZ=Asia/Shanghai date +%H:%M:%S)"
    sleep "${POLL_SEC}"
  done
  if [[ -f "${VIS}/fs10_fs5_fdg169_rerun_queue.pid" ]]; then
    pid=$(tr -d '[:space:]' < "${VIS}/fs10_fs5_fdg169_rerun_queue.pid" 2>/dev/null || true)
    while [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; do
      if _pgrep_real 'run_nnunet_psma_fs10|run_aligned_psma_fs10_fs5_pipeline_bg.sh'; then
        echo "[fc70-queue] waiting nnUNet fs10/fs5 rerun pid=${pid}…"
        sleep "${POLL_SEC}"
      else
        echo "[fc70-queue] rerun queue pid=${pid} idle — proceed"
        break
      fi
    done
  fi
}

echo "[fc70-queue] $(date '+%F %T') wait then launch fc70 pipeline"
_wait_prior
bash "${ROOT}/ICLR2026/run/run_aligned_psma_fc70_pipeline_bg.sh"
echo "[fc70-queue] fc70 done → chain PSMA fs0 / FDG TEST queue"
if ! pgrep -f "run_eval_fdg_shared_test20_bg.sh|run_eval_fdg_test20_bg.sh|run_fdg_eval_after_fc70|run_fdg20_test_after_fc70" >/dev/null 2>&1; then
  nohup bash "${ROOT}/ICLR2026/run/run_fdg_eval_after_fc70_queue_bg.sh" \
    >>"${VIS}/nohup_fdg_eval_queue.log" 2>&1 &
  echo $! > "${VIS}/fdg_eval_queue.pid"
  echo "[fc70-queue] fdg eval queue pid=$(cat "${VIS}/fdg_eval_queue.pid")"
fi
echo "[fc70-queue] ALL DONE"
