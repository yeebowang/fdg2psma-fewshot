#!/usr/bin/env bash
# Proto+Retrieval · PSMA fc70% (70%sup) → PSMA TEST20 (20%test), training-free.
#
#   bash ICLR2026/run/run_proto_retrieval_psma_fc70_test20_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"

STAMP="${STAMP:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_proto_retrieval_psma70_fc70_gpu013}"
GALLERY="${TASK1_PROTO_PSMA70_GALLERY:-${VIS}/proto_retrieval/psma70_gallery.npz}"
STRAT="${CTRL}/ICLR2026/data/splits_stratified_70_10_20.json"
TEST_JSON="${CTRL}/ICLR2026/data/splits_mae_psma_test20.json"
IMG="${DATA}/dataset1/imagesTr"
LAB="${DATA}/dataset1/labelsTr"
OUT_ROOT="${CTRL}/ICLR2026/runs/proto_retrieval/${STAMP}"
EVAL_ROOT="${OUT_ROOT}/psma_fc70_test20_eval"
TOPK="${TASK1_PROTO_TOPK:-3}"
PIPE_LOG="${VIS}/nohup_proto_retrieval_psma_fc70_${STAMP}.log"

exec > >(tee -a "${PIPE_LOG}") 2>&1
echo "[proto-fc70] STAMP=${STAMP} topk=${TOPK} gallery=${GALLERY}"

_dpy() {
  docker run --rm \
    -e PYTHONUNBUFFERED=1 \
    --user "$(id -u):$(id -g)" \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    -w "${CTRL}" \
    "${IMAGE}" \
    python3 "$@"
}

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --no-plot \
  --patch-json "{\"methods\":{\"proto_retrieval\":{\"psma_fc70\":{\"status\":\"running\",\"stamp\":\"${STAMP}\",\"training_free\":true,\"support_pool\":\"PSMA70%\",\"note\":\"PSMA70% retrieve → TEST20\"}}},\"updated_note\":\"Proto+ fc70 PSMA70%\"}" || true

if [[ ! -f "${GALLERY}" ]]; then
  echo "[proto-fc70] building PSMA70% gallery..."
  _dpy "${CTRL}/ICLR2026/scripts/proto_retrieval_build_gallery.py" \
    --pool psma70 \
    --stratified-json "${STRAT}" \
    --img-dir "${IMG}" \
    --lab-dir "${LAB}" \
    --out "${GALLERY}" \
    --workers "${TASK1_PROTO_GALLERY_WORKERS:-8}"
fi

mkdir -p "${EVAL_ROOT}"
PRED_DIR="${EVAL_ROOT}/predict"
OUT_JSON="${EVAL_ROOT}/test20_all.json"
echo "[proto-fc70] TEST20 eval (PSMA70% pool, topk=${TOPK})"
_dpy "${CTRL}/ICLR2026/scripts/proto_retrieval_eval_test20.py" \
  --gallery "${GALLERY}" \
  --cases-json "${TEST_JSON}" \
  --img-dir "${IMG}" \
  --lab-dir "${LAB}" \
  --pred-dir "${PRED_DIR}" \
  --out-json "${OUT_JSON}" \
  --fold 0 \
  --pool-mode psma70 \
  --topk "${TOPK}" \
  --stamp "${STAMP}" \
  --tag "proto_retrieval_psma70_fc70"

python3 - <<PY
import json
from pathlib import Path
from copy import deepcopy

d = json.loads(Path("${OUT_JSON}").read_text())
md = float(d.get("mean_dice_positive", d["mean_dice"]))
stamp = "${STAMP}"
summary = {
    "stamp": stamp,
    "method": "proto_retrieval",
    "split": "PSMA_TEST20",
    "support_pool": "PSMA70%",
    "topk": int("${TOPK}"),
    "mean": md,
    "mean_dice": md,
    "mean_dice_positive": md,
    "mean_dice_all_cases": d.get("mean_dice_all_cases"),
    "fp_rate": d.get("fp_rate"),
    "fn_rate": d.get("fn_rate"),
    "mean_fp": d.get("fp_rate"),
    "mean_fn": d.get("fn_rate"),
    "n_positive": d.get("n_positive"),
    "n_empty_gt": d.get("n_empty_gt"),
    "test_mean": md,
    "fold_test_dice": {"0": md},
    "metric": "TEST20 Dice/FP/FN; retrieve PSMA70% + prototype (empty GT excl. from Dice)",
    "protocol": "proto_retrieval_psma70_sup_psma20_test",
}
Path("${EVAL_ROOT}/aggregate_test20_fc70.json").write_text(json.dumps(summary, indent=2) + "\n")
Path("${VIS}/aggregate_proto_retrieval_psma_fc70_${STAMP}.json").write_text(json.dumps(summary, indent=2) + "\n")

board_p = Path("${BOARD}")
if board_p.is_file():
    b = json.loads(board_p.read_text())
    pr = b.setdefault("methods", {}).setdefault("proto_retrieval", {})
    st = pr.setdefault("psma_fc70", {})
    st.update({
        "status": "done",
        "stamp": stamp,
        "training_free": True,
        "support_pool": "PSMA70%",
        "fold_dice": {"0": md},
        "mean": md,
        "topk": int("${TOPK}"),
        "note": f"TEST20 DONE · PSMA70% · topk={int('${TOPK}')} · {md:.3f}",
        "metric": "TEST20 Dice; retrieve PSMA70% + prototype",
    })
    b["updated_note"] = f"Proto+ fc70 PSMA70% TEST20={md:.3f}"
    board_p.write_text(json.dumps(b, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true
echo "[proto-fc70] ALL DONE ${STAMP}"
