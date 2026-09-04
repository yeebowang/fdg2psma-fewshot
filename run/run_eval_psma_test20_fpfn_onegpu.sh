#!/usr/bin/env bash
# Re-eval MAE/MONAI PSMA TEST20 on a single idle GPU (folds sequential) → Dice/FP/FN.
# Used by gpu_idle_queue_scheduler. Does not fan out 3 folds onto 3 GPUs.
#
#   METHOD=mae|monai TASK1_PSMA_BOARD_STAGE=psma_fs50_f258 TASK1_CUDA_VISIBLE_DEVICES=0 \
#     bash ICLR2026/run/run_eval_psma_test20_fpfn_onegpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"

METHOD="${METHOD:-mae}"
GPU="${TASK1_CUDA_VISIBLE_DEVICES:-${TASK1_PSMA_FC70_GPU:-0}}"
GPU="${GPU%%,*}"
STAGE="${TASK1_PSMA_BOARD_STAGE:-psma_fs50_f258}"

case "${STAGE}" in
  psma_fs50_f258) N=50; FOLDS_DEFAULT=2,5,8 ;;
  psma_fs10_f258) N=10; FOLDS_DEFAULT=2,5,8 ;;
  psma_fs5_f258)  N=5;  FOLDS_DEFAULT=2,5,8 ;;
  *) echo "[error] unsupported stage ${STAGE}" >&2; exit 1 ;;
esac
case "${METHOD}" in
  mae) BOARD_KEY=mae_swinunetr ;;
  mae_scratch) BOARD_KEY=mae_scratch; FOLDS_DEFAULT=0,1,2,3,4,5,6,7,8 ;;
  monai_scratch) BOARD_KEY=monai_scratch; FOLDS_DEFAULT=0,1,2,3,4,5,6,7,8 ;;
  *) BOARD_KEY=monai_swinvit ;;
esac
N="${TASK1_FEWSHOT_N:-${N}}"
FOLDS_CSV="${TASK1_MAE_FEWSHOT_FOLDS_CSV:-${FOLDS_DEFAULT}}"

STAMP="${STAMP:-}"
if [[ -z "${STAMP}" ]]; then
  STAMP="$(python3 - <<PY
import json
from pathlib import Path
b = json.loads(Path("${BOARD}").read_text())
st = (b.get("methods") or {}).get("${BOARD_KEY}", {}).get("${STAGE}") or {}
print(st.get("stamp") or "")
PY
)"
fi
[[ -n "${STAMP}" ]] || { echo "[error] no STAMP for ${BOARD_KEY}.${STAGE}" >&2; exit 1; }

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"updated_note\":\"${METHOD} ${STAGE} · RUNNING (GPU ${GPU}) FP/FN re-eval\",\"methods\":{\"${BOARD_KEY}\":{\"${STAGE}\":{\"status\":\"running\",\"device\":\"gpu\",\"gpu_ids\":\"${GPU}\",\"note\":\"RUNNING (GPU ${GPU}) · FP/FN re-eval\"}}}}" \
  || true

echo "[fpfn-onegpu] METHOD=${METHOD} STAGE=${STAGE} STAMP=${STAMP} GPU=${GPU} folds=${FOLDS_CSV}"
IFS=',' read -r -a FOLD_ARR <<< "${FOLDS_CSV}"
for fold in "${FOLD_ARR[@]}"; do
  fold="${fold// /}"
  [[ -n "${fold}" ]] || continue
  echo "[fpfn-onegpu] fold${fold} → GPU${GPU}"
  METHOD="${METHOD}" STAMP="${STAMP}" \
    TASK1_FEWSHOT_N="${N}" \
    TASK1_PSMA_BOARD_STAGE="${STAGE}" \
    TASK1_MAE_FEWSHOT_FOLDS_CSV="${fold}" \
    TASK1_FOLD_GPUS="${fold}:${GPU}" \
    TASK1_TEST_SKIP_DONE=0 \
    bash "${CTRL}/ICLR2026/run/run_eval_psma_test20_f258_bg.sh"
done

echo "[fpfn-onegpu] ALL DONE ${METHOD} ${STAGE} ${STAMP}"
