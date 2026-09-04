#!/usr/bin/env bash
# Aligned board queue for hemingduo / chenyixin (scratch + Dataset619 pretrained).
#
# Policy (hard):
#   - scratch rows: FDG (tr70/val0 · 169ep) → PSMA fs50/10/5 / fs0 / fc70 / FDG TEST
#   - pretrained rows: load ONLY Dataset619 MultiTalent (Zenodo 13753413) BEFORE FDG, then same cascade
#   - NEVER use GC final submission weights for board scoring:
#       Zenodo 14007247 LesionTracer, BIRTH EDT, YixinChen LocalEdit/TACE
#
# Usage:
#   # download pretrain first (pretrained rows)
#   bash ICLR2026/run/download_dataset619_multitalent_pretrain_bg.sh
#   # then queue (order: scratch → pretrained, hemingduo → chenyixin)
#   bash ICLR2026/run/run_competition_aligned_fdg_psma_queue_bg.sh
#
# Env:
#   TASK1_COMP_METHODS="hemingduo_scratch hemingduo chenyixin_scratch chenyixin"
#   TASK1_COMP_SKIP_DOWNLOAD=1   # if Dataset619 already present
#   TASK1_COMP_DRY_RUN=1        # policy + board patch only, no train
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ICLR_VIS="${ROOT}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
LOG="${ICLR_VIS}/nohup_competition_aligned_fdg_psma_queue.log"
mkdir -p "${ICLR_VIS}"
exec > >(tee -a "${LOG}") 2>&1

METHODS=(${TASK1_COMP_METHODS:-hemingduo_scratch hemingduo chenyixin_scratch chenyixin})
DRY="${TASK1_COMP_DRY_RUN:-0}"

echo "[comp-aligned] $(date '+%F %T') methods=${METHODS[*]} dry=${DRY}"

# Refresh board notes / order (scratch before pretrained; forbid final ckpts)
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --png "${ICLR_VIS}/progress_iclr2026_aligned_fdg_fs50_f258_board.png" \
  --patch-json '{"updated_note":"competition: scratch=FDG→PSMA; pretrained=Dataset619→FDG→PSMA; forbid GC final ckpts"}' || true

# Download Dataset619 if any pretrained method is requested
need_619=0
for m in "${METHODS[@]}"; do
  if [[ "${m}" == "hemingduo" || "${m}" == "chenyixin" ]]; then
    need_619=1
  fi
done
if [[ "${need_619}" == "1" && "${TASK1_COMP_SKIP_DOWNLOAD:-0}" != "1" ]]; then
  bash "${ROOT}/ICLR2026/run/download_dataset619_multitalent_pretrain_bg.sh"
fi
if [[ "${need_619}" == "1" ]]; then
  python3 "${ROOT}/ICLR2026/scripts/assert_competition_board_weights.py" --require-dataset619
  PRETRAIN_CKPT="$(tr -d '[:space:]' < "${ROOT}/ICLR2026/weights/Dataset619_nativemultistem/PRETRAIN_CHECKPOINT.txt")"
  echo "[comp-aligned] Dataset619 pretrain=${PRETRAIN_CKPT}"
else
  python3 "${ROOT}/ICLR2026/scripts/assert_competition_board_weights.py"
  PRETRAIN_CKPT=""
fi

if [[ "${DRY}" == "1" ]]; then
  echo "[comp-aligned] DRY_RUN=1 → stop before training"
  exit 0
fi

echo "[comp-aligned] WARN: MultiTalent organ dual-head FDG trainer not yet wired for Dataset228/221."
echo "[comp-aligned]        Do NOT fall back to LesionTracer final / EDT / LocalEdit for board cells."
echo "[comp-aligned]        Next: wire autoPET3_Trainer (+ organ labels) or ResEncL MultiTalent plans on FDG,"
echo "[comp-aligned]        then chain PSMA like run_nnunet_aligned_fdg169_psma_f258_bg.sh per method."
echo "[comp-aligned] DONE policy gate $(date '+%F %T')"
exit 0
