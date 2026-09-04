#!/usr/bin/env bash
# CPU-only: Proto+ fs50/fs10/fs5/fs0 share one FDG100% TEST20 — rescore Dice/FP/FN once.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
LOG="${VIS}/nohup_rescore_proto_fewshot_fp_fn.log"
PID_FILE="${VIS}/rescore_proto_fewshot_fp_fn.pid"

if [[ -f "${PID_FILE}" ]]; then
  old="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[proto-cpu] already running pid=${old}"
    exit 0
  fi
fi

# Mark 4 columns RUNNING (CPU) immediately
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" \
  --patch-json '{"updated_note":"CPU · Proto fs50/10/5/0 rescore Dice/FP/FN (shared once)",
    "methods":{"proto_retrieval":{
      "psma_fs50_f258":{"status":"running","device":"cpu","note":"RUNNING (CPU) · rescore FP/FN once"},
      "psma_fs10_f258":{"status":"running","device":"cpu","note":"RUNNING (CPU) · same as fs50"},
      "psma_fs5_f258":{"status":"running","device":"cpu","note":"RUNNING (CPU) · same as fs50"},
      "psma_fs0":{"status":"running","device":"cpu","note":"RUNNING (CPU) · same as fs50"}
    }}}' || true

nohup bash -c "
  echo \$\$ > '${PID_FILE}'
  echo \"[proto-cpu] \$(date '+%F %T') start\"
  python3 '${ROOT}/ICLR2026/scripts/rescore_board_dice_fp_fn.py' --board '${BOARD}' --proto-fewshot-only
  python3 '${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py' --board '${BOARD}' \
    --patch-json '{\"updated_note\":\"CPU DONE · Proto fs50/10/5/0 Dice/FP/FN\"}' || true
  rm -f '${PID_FILE}'
  echo \"[proto-cpu] \$(date '+%F %T') DONE\"
" >>"${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "[proto-cpu] started pid=$(cat "${PID_FILE}") log=${LOG}"
