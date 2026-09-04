#!/usr/bin/env bash
# CPU now: SegAnyPET scratch PSMA fs0 + FDG TEST FP/FN from existing nifti.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
LOG="${VIS}/nohup_rescore_seganypet_scratch_tail.log"
PID_FILE="${VIS}/rescore_seganypet_scratch_tail.pid"
mkdir -p "${VIS}"
if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[rescore-scratch] already running pid=${old}"
    exit 0
  fi
fi
nohup bash -c "
set -euo pipefail
python3 '${ROOT}/ICLR2026/scripts/rescore_board_dice_fp_fn.py' \
  --board '${BOARD}' --seganypet-scratch-tail
python3 '${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py' --board '${BOARD}' || true
echo '[rescore-scratch] DONE'
" >>"${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "[rescore-scratch] started pid=$(cat "${PID_FILE}") log=${LOG}"
