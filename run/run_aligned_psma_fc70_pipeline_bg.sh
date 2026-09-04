#!/usr/bin/env bash
# PSMA fc70% pipeline: all trainable methods · single run · FDG init · decline → TEST20
#   bash ICLR2026/run/run_aligned_psma_fc70_pipeline_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
METHODS="${TASK1_METHODS:-nnunet,mae,monai,dpdnet,seganypet}"
PIPE_LOG="${VIS}/nohup_aligned_psma_fc70_pipeline.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[fc70-pipeline] methods=${METHODS} · 421/59 train70 · tr25 val25 bs2 · single run"

_fc70_should_run() {
  local mkey="$1"
  python3 - <<PY
import json, sys
from pathlib import Path
b = json.loads(Path("${BOARD}").read_text())
st = (b.get("methods") or {}).get("${mkey}", {}).get("psma_fc70") or {}
s = (st.get("status") or "pending").lower()
# skip if already done or running (e.g. parallel launch on idle GPU)
if s in ("done", "running"):
    sys.exit(1)
# incomplete fc70 yields to extra-fold 9fold
done = Path("${VIS}/TASK1_PSMA_EXTRA_FOLDS_9FOLD_DONE.txt")
extra_done = done.is_file() and "status=ok" in done.read_text(encoding="utf-8", errors="ignore")
stamp = (st.get("stamp") or "").strip()
if stamp and not extra_done:
    sys.exit(1)
sys.exit(0)
PY
}

_fc70_skip_or_run() {
  local mkey="$1" label="$2" script="$3"
  if _fc70_should_run "${mkey}"; then
    echo "[fc70-pipeline] start ${label}"
    bash "${script}"
  else
    echo "[fc70-pipeline] skip ${label} (board psma_fc70 already running/done)"
  fi
}

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"updated_note\":\"fc70% PSMA pipeline starting\",\"queue\":[\"fc70: ${METHODS}\"]}" || true

bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true

if [[ "${METHODS}" == *nnunet* || "${METHODS}" == "all" ]]; then
  _fc70_skip_or_run nnunet "nnUNet fc70" "${ROOT}/ICLR2026/run/run_nnunet_psma_fc70_decline_and_test_bg.sh"
fi
if [[ "${METHODS}" == *mae* || "${METHODS}" == "all" ]]; then
  bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
  _fc70_skip_or_run mae_swinunetr "MAE fc70" "${ROOT}/ICLR2026/run/run_mae_psma_fc70_from_fdg_seg_bg.sh"
fi
if [[ "${METHODS}" == *monai* || "${METHODS}" == "all" ]]; then
  bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
  _fc70_skip_or_run monai_swinvit "MONAI fc70" "${ROOT}/ICLR2026/run/run_monai_psma_fc70_from_fdg_seg_bg.sh"
fi
if [[ "${METHODS}" == *dpdnet* || "${METHODS}" == "all" ]]; then
  bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
  _fc70_skip_or_run dpdnet "DpDNet fc70" "${ROOT}/ICLR2026/run/run_dpdnet_psma_fc70_decline_and_test_bg.sh"
fi
if [[ "${METHODS}" == *seganypet* || "${METHODS}" == "all" ]]; then
  _fc70_skip_or_run seganypet "SegAnyPET fc70" "${ROOT}/ICLR2026/run/run_seganypet_psma_fc70_from_fdg_bg.sh"
fi

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json '{"queue":[],"updated_note":"fc70% PSMA pipeline done"}' || true
echo "[fc70-pipeline] ALL DONE"
