#!/usr/bin/env bash
# Proto+Retrieval PSMA100% only · TEST20 comparison (board stays FDG-only).
#
#   bash ICLR2026/run/run_proto_retrieval_psma100_test20_f258_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"

STAMP="${STAMP:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_proto_retrieval_psma100_f258_gpu013}"
GALLERY="${TASK1_PROTO_PSMA_GALLERY:-${CTRL}/ICLR2026/data/proto_retrieval/psma100_gallery.npz}"
STRAT="${CTRL}/ICLR2026/data/splits_stratified_70_10_20.json"
TEST_JSON="${CTRL}/ICLR2026/data/splits_mae_psma_test20.json"
IMG="${DATA}/dataset1/imagesTr"
LAB="${DATA}/dataset1/labelsTr"
OUT_ROOT="${CTRL}/ICLR2026/runs/proto_retrieval/${STAMP}"
EVAL_ROOT="${OUT_ROOT}/psma_test20_eval"
TOPK="${TASK1_PROTO_TOPK:-3}"
FOLDS="${TASK1_FOLDS:-2,5,8}"
PIPE_LOG="${VIS}/nohup_proto_retrieval_psma100_${STAMP}.log"

exec > >(tee -a "${PIPE_LOG}") 2>&1
echo "[proto-psma100] STAMP=${STAMP} topk=${TOPK}"

_dpy() {
  docker run --rm \
    -e PYTHONUNBUFFERED=1 \
    --user "$(id -u):$(id -g)" \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    -w "${CTRL}" \
    "${IMAGE}" \
    python3 "$@"
}

if [[ ! -f "${GALLERY}" ]]; then
  echo "[proto-psma100] building PSMA100% gallery..."
  _dpy "${CTRL}/ICLR2026/scripts/proto_retrieval_build_gallery.py" \
    --pool psma100 \
    --stratified-json "${STRAT}" \
    --img-dir "${IMG}" \
    --lab-dir "${LAB}" \
    --out "${GALLERY}" \
    --workers "${TASK1_PROTO_GALLERY_WORKERS:-8}"
fi

mkdir -p "${EVAL_ROOT}"
PRED_DIR="${EVAL_ROOT}/predict"
OUT_JSON="${EVAL_ROOT}/test20_all.json"
echo "[proto-psma100] TEST20 eval (PSMA100% pool, topk=${TOPK})"
_dpy "${CTRL}/ICLR2026/scripts/proto_retrieval_eval_test20.py" \
  --gallery "${GALLERY}" \
  --cases-json "${TEST_JSON}" \
  --img-dir "${IMG}" \
  --lab-dir "${LAB}" \
  --pred-dir "${PRED_DIR}" \
  --out-json "${OUT_JSON}" \
  --fold 0 \
  --pool-mode psma100 \
  --topk "${TOPK}" \
  --stamp "${STAMP}" \
  --tag "proto_retrieval_psma100"

IFS=',' read -r -a FOLD_ARR <<< "${FOLDS}"
for fold in "${FOLD_ARR[@]}"; do
  mkdir -p "${EVAL_ROOT}/fold${fold}/predict"
  cp -a "${OUT_JSON}" "${EVAL_ROOT}/fold${fold}_test20.json"
  cp -a "${PRED_DIR}/." "${EVAL_ROOT}/fold${fold}/predict/" 2>/dev/null || true
done

python3 - <<PY
import json, statistics
from pathlib import Path
eval_root = Path("${EVAL_ROOT}")
vis = Path("${VIS}")
stamp = "${STAMP}"
d = json.loads((eval_root / "test20_all.json").read_text())
fd = {str(f): d["mean_dice"] for f in [int(x) for x in "${FOLDS}".split(",")]}
summary = {
    "stamp": stamp,
    "method": "proto_retrieval",
    "variant": "PSMA100% only (comparison)",
    "split": "PSMA_TEST20",
    "support_pool": "PSMA100%",
    "topk": int("${TOPK}"),
    "fold_test_dice": fd,
    "test_mean": d["mean_dice"],
    "test_std": 0.0,
    "metric": "TEST20 Dice; retrieve PSMA100% + prototype",
    "board_note": "comparison only; board keeps FDG100% only",
    "fdg100_baseline_mean": 0.10042805478734147,
}
(eval_root / "aggregate_test20_f258.json").write_text(json.dumps(summary, indent=2)+"\n")
(vis / f"aggregate_proto_retrieval_psma100_test20_f258_{stamp}.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2))
PY

echo "[proto-psma100] DONE ${STAMP} (comparison aggregate only, board unchanged)"
