#!/usr/bin/env bash
# Refresh aligned FDG→PSMA f258 progress board JSON + PNG every N seconds.
#
#   bash ICLR2026/run/run_aligned_board_watch_bg.sh
#   TASK1_BOARD_WATCH_SEC=60 bash ICLR2026/run/run_aligned_board_watch_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"
BOARD_PNG="${TASK1_ALIGN_BOARD_PNG:-${LOG_DIR}/progress_iclr2026_aligned_fdg_fs50_f258_board.png}"
WATCH_SEC="${TASK1_BOARD_WATCH_SEC:-30}"
PID_FILE="${LOG_DIR}/aligned_board_watch.pid"
LOG="${LOG_DIR}/nohup_aligned_board_watch.log"

if [[ -f "${PID_FILE}" ]]; then
  old="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[board-watch] already running pid=${old} (stop: kill ${old})"
    exit 0
  fi
fi

nohup python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --png "${BOARD_PNG}" \
  --watch "${WATCH_SEC}" \
  >>"${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "[board-watch] pid=$(cat "${PID_FILE}") every=${WATCH_SEC}s board=${BOARD_JSON} log=${LOG}"
