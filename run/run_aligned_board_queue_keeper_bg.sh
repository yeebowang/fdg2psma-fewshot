#!/usr/bin/env bash
# Keep aligned pipeline queue workers alive; restart dead workers (self-check companion).
#
#   bash ICLR2026/run/run_aligned_board_queue_keeper_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
INTERVAL="${TASK1_QUEUE_KEEPER_SEC:-120}"
LOG="${VIS}/nohup_aligned_board_queue_keeper.log"
PID_FILE="${VIS}/aligned_board_queue_keeper.pid"

if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[queue-keeper] already running pid=${old}"
    exit 0
  fi
fi

nohup bash -c "
set -euo pipefail
exec >>'${LOG}' 2>&1
echo '[queue-keeper] start \$(date)'
while true; do
  bash '${ROOT}/ICLR2026/run/run_aligned_board_watch_bg.sh' || true
  bash '${ROOT}/ICLR2026/run/run_gpu_idle_queue_scheduler_bg.sh' || true
  python3 '${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py' \
    --board '${BOARD}' --no-plot \
    --patch-json '{\"updated_note\":\"queue keeper refresh\"}' || true
  dead=\$(python3 - <<'PY'
import json
from pathlib import Path
b = json.loads(Path('${BOARD}').read_text())
print(','.join((b.get('health_check') or {}).get('dead_workers') or []))
PY
)
  if [[ -n \"\${dead}\" ]]; then
    echo \"[queue-keeper] dead workers: \${dead}\"
    if [[ \"\${dead}\" == *fs10_fs5* ]] && ! pgrep -f 'run_aligned_psma_fs10_fs5_pipeline' >/dev/null; then
      nohup bash '${ROOT}/ICLR2026/run/run_aligned_psma_fs10_fs5_pipeline_bg.sh' >>'${VIS}/nohup_aligned_psma_fs10_fs5_pipeline.log' 2>&1 &
      echo \$! > '${VIS}/aligned_psma_fs10_fs5_pipeline.pid'
    fi
    if [[ \"\${dead}\" == *nnunet_fs10_fs5* ]] && ! pgrep -f 'run_nnunet_psma_fs10_fs5_rerun' >/dev/null; then
      nohup bash '${ROOT}/ICLR2026/run/run_nnunet_psma_fs10_fs5_rerun_fdg169_after_methods_queue_bg.sh' >>'${VIS}/nohup_nnunet_fs10_fs5_rerun_fdg169_queue.log' 2>&1 &
      echo \$! > '${VIS}/fs10_fs5_fdg169_rerun_queue.pid'
    fi
    if [[ \"\${dead}\" == *fc70_queue* ]] && ! pgrep -f 'run_psma_fc70_after_main_queue' >/dev/null; then
      extra_done=0
      if [[ -f '${VIS}/TASK1_PSMA_EXTRA_FOLDS_9FOLD_DONE.txt' ]] && grep -q 'status=ok' '${VIS}/TASK1_PSMA_EXTRA_FOLDS_9FOLD_DONE.txt'; then
        extra_done=1
      fi
      if [[ \"\${extra_done}\" -eq 1 ]]; then
        nohup bash '${ROOT}/ICLR2026/run/run_psma_fc70_after_main_queue_bg.sh' >>'${VIS}/nohup_psma_fc70_queue.log' 2>&1 &
        echo \$! > '${VIS}/psma_fc70_queue.pid'
      else
        echo \"[queue-keeper] skip fc70_queue restart until extra-fold 9fold done\"
      fi
    fi
    if [[ \"\${dead}\" == *eval_queue* ]] && ! pgrep -af 'run_eval_fdg_shared_test20_bg.sh|run_eval_fdg_test20_bg.sh|run_fdg_eval_after_fc70_queue_bg.sh' 2>/dev/null | grep -Ev 'queue_keeper' | grep -q .; then
      nohup bash '${ROOT}/ICLR2026/run/run_fdg_eval_after_fc70_queue_bg.sh' >>'${VIS}/nohup_fdg_eval_queue.log' 2>&1 &
      echo \$! > '${VIS}/fdg_eval_queue.pid'
    fi
    if [[ \"\${dead}\" == *extra_folds_9fold* ]] && ! pgrep -f 'run_psma_extra_folds_9fold_after_eval|run_aligned_psma_extra_folds_9fold' >/dev/null; then
      nohup bash '${ROOT}/ICLR2026/run/run_psma_extra_folds_9fold_after_eval_queue_bg.sh' >>'${VIS}/nohup_extra_folds_9fold_queue.log' 2>&1 &
      echo \$! > '${VIS}/extra_folds_9fold_queue.pid'
    fi
    if [[ \"\${dead}\" == *mae_scratch_9fold* ]] && ! pgrep -f 'run_mae_scratch_after_extra_folds_queue|run_aligned_mae_scratch_9fold_pipeline' >/dev/null; then
      nohup bash '${ROOT}/ICLR2026/run/run_mae_scratch_after_extra_folds_queue_bg.sh' >>'${VIS}/nohup_mae_scratch_9fold_queue.log' 2>&1 &
      echo \$! > '${VIS}/mae_scratch_9fold_queue.pid'
    fi
    if [[ \"\${dead}\" == *monai_scratch_9fold* ]] && ! pgrep -f 'run_aligned_monai_scratch_9fold_pipeline' >/dev/null; then
      nohup bash '${ROOT}/ICLR2026/run/run_aligned_monai_scratch_9fold_pipeline_bg.sh' >>'${VIS}/nohup_monai_scratch_9fold_pipeline.log' 2>&1 &
      echo \$! > '${VIS}/monai_scratch_9fold_pipeline.pid'
    fi
    if [[ \"\${dead}\" == *seganypet_scratch_9fold* ]] && ! pgrep -f 'run_aligned_seganypet_scratch_9fold_pipeline' >/dev/null; then
      nohup bash '${ROOT}/ICLR2026/run/run_aligned_seganypet_scratch_9fold_pipeline_bg.sh' >>'${VIS}/nohup_seganypet_scratch_9fold_pipeline.log' 2>&1 &
      echo \$! > '${VIS}/seganypet_scratch_9fold_pipeline.pid'
    fi
  fi
  sleep ${INTERVAL}
done
" >>"${LOG}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[queue-keeper] pid=$(cat "${PID_FILE}") interval=${INTERVAL}s log=${LOG}"
