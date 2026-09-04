#!/usr/bin/env bash
# Queue both competition scratch rows (no Dataset619): hemingduo_scratch → chenyixin_scratch.
#
#   bash ICLR2026/run/run_competition_scratch_queue_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ICLR_VIS="${ROOT}/ICLR2026/vis"
LOG="${ICLR_VIS}/nohup_competition_scratch_queue.log"
mkdir -p "${ICLR_VIS}"
exec > >(tee -a "${LOG}") 2>&1

METHODS=(${TASK1_COMP_SCRATCH_METHODS:-hemingduo_scratch chenyixin_scratch})
echo "[comp-scratch-queue] $(date '+%F %T') methods=${METHODS[*]}"

python3 "${ROOT}/ICLR2026/scripts/assert_competition_board_weights.py" || exit 1

for m in "${METHODS[@]}"; do
  echo "[comp-scratch-queue] === start ${m} $(date '+%F %T') ==="
  TASK1_BOARD_METHOD="${m}" bash "${ROOT}/ICLR2026/run/run_competition_scratch_aligned_fdg_psma_bg.sh"
  echo "[comp-scratch-queue] === done ${m} $(date '+%F %T') ==="
done

echo "[comp-scratch-queue] ALL DONE $(date '+%F %T')"
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json" \
  --png "${ICLR_VIS}/progress_iclr2026_aligned_fdg_fs50_f258_board.png" \
  --patch-json '{"updated_note":"competition scratch queue done (hemingduo_scratch→chenyixin_scratch)"}' || true
