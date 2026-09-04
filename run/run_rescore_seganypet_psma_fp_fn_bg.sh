#!/usr/bin/env bash
# CPU-only: SegAnyPET PSMA TEST fold*_pred → Dice (empty-GT excl.) + voxel FP/FN.
# Default: fs50/fs10/fs5 (preds ready). Pass STAGES=psma_fc70 when TEST preds exist.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
STAGES="${STAGES:-psma_fs50_f258,psma_fs10_f258,psma_fs5_f258}"
LOG="${VIS}/nohup_rescore_seganypet_psma_fp_fn.log"
PID_FILE="${VIS}/rescore_seganypet_psma_fp_fn.pid"

if [[ -f "${PID_FILE}" ]]; then
  old="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[seganypet-cpu] already running pid=${old}"
    exit 0
  fi
fi

# Mark stages RUNNING (CPU) without touching fc70 train
python3 - <<PY
import json
from pathlib import Path
board_p = Path("${BOARD}")
board = json.loads(board_p.read_text())
seg = board.setdefault("methods", {}).setdefault("seganypet", {})
for stage in "${STAGES}".split(","):
    stage = stage.strip()
    if not stage:
        continue
    st = seg.setdefault(stage, {})
    # do not clobber live GPU train (fc70) unless it already has TEST preds / done mean
    if stage == "psma_fc70" and (st.get("status") or "").lower() == "running" and st.get("mean") is None:
        print(f"[skip-mark] {stage} still training")
        continue
    st["status"] = "running"
    st["device"] = "cpu"
    st["note"] = "RUNNING (CPU) · rescore Dice/FP/FN from pred"
board["updated_note"] = "CPU · SegAnyPET rescore Dice/FP/FN"
board_p.write_text(json.dumps(board, indent=2) + "\n")
print("[mark] stages=${STAGES}")
PY
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" --plot-only || true

nohup bash -c "
  echo \$\$ > '${PID_FILE}'
  echo \"[seganypet-cpu] \$(date '+%F %T') start stages=${STAGES}\"
  python3 '${ROOT}/ICLR2026/scripts/rescore_board_dice_fp_fn.py' --board '${BOARD}' \\
    --seganypet-only --seganypet-stages '${STAGES}'
  python3 '${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py' --board '${BOARD}' --plot-only || true
  python3 '${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py' --board '${BOARD}' \\
    --patch-json '{\"updated_note\":\"CPU DONE · SegAnyPET Dice/FP/FN\"}' || true
  rm -f '${PID_FILE}'
  echo \"[seganypet-cpu] \$(date '+%F %T') DONE\"
" >>"${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "[seganypet-cpu] started pid=$(cat "${PID_FILE}") log=${LOG} stages=${STAGES}"
