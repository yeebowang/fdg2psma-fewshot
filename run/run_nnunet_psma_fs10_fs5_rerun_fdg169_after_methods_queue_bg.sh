#!/usr/bin/env bash
# Queue nnUNet PSMA fs10+fs5 rerun (FDG169 checkpoint_final) after fs10/fs5 methods pipeline.
# DpDNet uses dpdnet_fdg_LAST_STAMP.txt at train time — no separate rerun needed.
#
#   bash ICLR2026/run/run_nnunet_psma_fs10_fs5_rerun_fdg169_after_methods_queue_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
PIPE_LOG="${TASK1_WAIT_PIPELINE_LOG:-${VIS}/nohup_aligned_psma_fs10_fs5_pipeline.log}"
DONE_MARK="${VIS}/TASK1_FS10_FS5_METHODS_PIPELINE_DONE.txt"
POLL_SEC="${TASK1_CHAIN_POLL_SEC:-60}"
WAIT_PID="${TASK1_WAIT_PIPELINE_PID:-}"

NN_FDG_STAMP="${TASK1_UDA_FDG_STAMP:-20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep}"
NN_FDG_BEST="${TASK1_UDA_FDG_BEST:-/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth}"

QUEUE_LOG="${VIS}/nohup_nnunet_fs10_fs5_rerun_fdg169_queue.log"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

if [[ -z "${WAIT_PID}" ]]; then
  WAIT_PID="$(pgrep -f "run_aligned_psma_fs10_fs5_pipeline_bg.sh" 2>/dev/null | head -1 || true)"
fi

echo "[nnunet-rerun-queue] $(date '+%F %T') wait methods pipeline pid=${WAIT_PID:-none}"
echo "[nnunet-rerun-queue] nnUNet FDG=${NN_FDG_BEST}"
echo "[nnunet-rerun-queue] DpDNet in-flight → official via dpdnet_fdg_LAST_STAMP.txt (no rerun)"

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "$(python3 - <<PY
import json
print(json.dumps({
  "updated_note": "nnUNet fs10/fs5 FDG169 rerun queued; DpDNet in-flight uses LAST_STAMP 169ep",
  "queue": [
    "monai/dpdnet/seganypet psma fs10+fs5 (in flight)",
    "nnunet.psma_fs10_f258 rerun FDG169 final",
    "nnunet.psma_fs5_f258 rerun FDG169 final",
  ],
}))
PY
)" || true

_methods_done() {
  [[ -f "${DONE_MARK}" ]] && grep -q 'status=ok' "${DONE_MARK}" && return 0
  [[ -f "${PIPE_LOG}" ]] && grep -q '\[pipeline\] ALL DONE fs=10,5' "${PIPE_LOG}" && return 0
  if [[ -f "${PIPE_LOG}" ]] \
     && grep -q '\[seg-decline\] ALL DONE fs10' "${PIPE_LOG}" \
     && grep -q '\[seg-decline\] ALL DONE fs5' "${PIPE_LOG}" \
     && grep -q '\[mae-fdgseg-fs10\] ALL DONE' "${PIPE_LOG}" \
     && grep -q '\[mae-fdgseg-fs5\] ALL DONE' "${PIPE_LOG}" \
     && grep -q '\[monai-fdgseg-fs10\] ALL DONE' "${PIPE_LOG}" \
     && grep -q '\[monai-fdgseg-fs5\] ALL DONE' "${PIPE_LOG}" \
     && grep -q '\[dpd-decline\] ALL DONE fs10' "${PIPE_LOG}" \
     && grep -q '\[dpd-decline\] ALL DONE fs5' "${PIPE_LOG}"; then
    return 0
  fi
  return 1
}

_board_nnunet_fs10_fs5_done() {
  python3 - <<PY
import json
from pathlib import Path
board = Path("${BOARD}")
if not board.is_file():
    raise SystemExit(1)
b = json.loads(board.read_text())
nn = (b.get("methods") or {}).get("nnunet") or {}
ok = all((nn.get(s) or {}).get("status") == "done" for s in ("psma_fs10_f258", "psma_fs5_f258"))
raise SystemExit(0 if ok else 1)
PY
}

while true; do
  if _methods_done; then
    echo "[nnunet-rerun-queue] methods pipeline done"
    break
  fi
  if [[ -n "${WAIT_PID}" ]] && ! kill -0 "${WAIT_PID}" 2>/dev/null; then
    _methods_done && break
  fi
  echo "[nnunet-rerun-queue] waiting… $(TZ=Asia/Shanghai date +%H:%M:%S)"
  sleep "${POLL_SEC}"
done

if _board_nnunet_fs10_fs5_done; then
  echo "[nnunet-rerun-queue] skip rerun — board nnUNet fs10/fs5 already done"
  python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD}" --no-plot \
    --patch-json '{"queue":[],"updated_note":"nnUNet fs10/fs5 skip rerun (board done)"}' || true
  echo "[nnunet-rerun-queue] ALL DONE skip $(date '+%F %T')"
  exit 0
fi

bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true

echo "[nnunet-rerun-queue] launch nnUNet fs10+fs5 · FDG169 final"
export TASK1_METHODS=nnunet TASK1_FEWSHOT_LIST=10,5
export TASK1_UDA_FDG_STAMP="${NN_FDG_STAMP}" TASK1_UDA_FDG_BEST="${NN_FDG_BEST}"
export TASK1_ALIGN_BOARD_JSON="${BOARD}"
bash "${ROOT}/ICLR2026/run/run_aligned_psma_fs10_fs5_pipeline_bg.sh"

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json '{"queue":[],"updated_note":"nnUNet fs10/fs5 FDG169 rerun done"}' || true

echo "[nnunet-rerun-queue] ALL DONE $(date '+%F %T')"
