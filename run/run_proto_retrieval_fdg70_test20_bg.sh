#!/usr/bin/env bash
# Proto+Retrieval · FDG70% sup → FDG20% TEST (training-free).
# Prefer this over the FDG80% path; writes aggregate so METHOD=all can skip.
#
#   bash ICLR2026/run/run_proto_retrieval_fdg70_test20_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
AGG_DIR="${VIS}/fdg_test20"
EVAL_ROOT="${WORK}/fdg_test20_eval/proto_retrieval"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"
STRAT="${CTRL}/ICLR2026/data/splits_stratified_70_10_20.json"
TEST_JSON="${CTRL}/ICLR2026/data/splits_fdg_test20.json"
IMG="${DATA}/dataset1/imagesTr"
LAB="${DATA}/dataset1/labelsTr"
GALLERY="${TASK1_PROTO_FD70_GALLERY:-${VIS}/proto_retrieval/fdg70_gallery.npz}"
STAMP="${STAMP:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_proto_fdg70_test20}"
TOPK="${TASK1_PROTO_TOPK:-3}"
PIPE_LOG="${VIS}/nohup_proto_retrieval_fdg70_test20_${STAMP}.log"

mkdir -p "${AGG_DIR}" "${EVAL_ROOT}" "$(dirname "${GALLERY}")"
exec > >(tee -a "${PIPE_LOG}") 2>&1
echo "[proto-fdg70] STAMP=${STAMP} gallery=${GALLERY}"

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
  --patch-json "{\"methods\":{\"proto_retrieval\":{\"fdg_test20\":{\"status\":\"running\",\"stamp\":\"${STAMP}\",\"training_free\":true,\"support_pool\":\"FDG70%\",\"note\":\"FDG70% → FDG20% TEST\"}}},\"updated_note\":\"Proto+ FDG70% TEST\"}" || true

if [[ ! -f "${GALLERY}" ]]; then
  echo "[proto-fdg70] building FDG70% gallery..."
  _dpy "${CTRL}/ICLR2026/scripts/proto_retrieval_build_gallery.py" \
    --pool fdg70 \
    --stratified-json "${STRAT}" \
    --img-dir "${IMG}" \
    --lab-dir "${LAB}" \
    --out "${GALLERY}" \
    --workers "${TASK1_PROTO_GALLERY_WORKERS:-8}"
fi

OUT_JSON="${EVAL_ROOT}/fdg_test20_all.json"
_dpy "${CTRL}/ICLR2026/scripts/proto_retrieval_eval_test20.py" \
  --gallery "${GALLERY}" \
  --cases-json "${TEST_JSON}" \
  --img-dir "${IMG}" \
  --lab-dir "${LAB}" \
  --pred-dir "${EVAL_ROOT}/predict" \
  --out-json "${OUT_JSON}" \
  --fold 0 \
  --pool-mode fdg70 \
  --topk "${TOPK}" \
  --stamp "${STAMP}" \
  --tag "proto_fdg70_sup_fdg20_test"

python3 - <<PY
import json
from pathlib import Path
d = json.loads(Path("${OUT_JSON}").read_text())
md = float(d.get("mean_dice_positive", d["mean_dice"]))
agg = {
    "method": "proto_retrieval",
    "mean_dice": md,
    "mean_dice_positive": md,
    "mean_dice_all_cases": d.get("mean_dice_all_cases"),
    "fp_rate": d.get("fp_rate"),
    "fn_rate": d.get("fn_rate"),
    "mean_fp": d.get("fp_rate"),
    "mean_fn": d.get("fn_rate"),
    "mean": md,
    "n_scored": d.get("n_scored") or d.get("n_cases"),
    "n_positive": d.get("n_positive"),
    "n_empty_gt": d.get("n_empty_gt"),
    "support_pool": "FDG70%",
    "test_split": "FDG20%",
    "topk": int("${TOPK}"),
    "eval_stamp": "${STAMP}",
    "protocol": "proto_retrieval_fdg70_sup_fdg20_test",
}
Path("${EVAL_ROOT}/aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
Path("${AGG_DIR}/aggregate_proto_retrieval.json").write_text(json.dumps(agg, indent=2) + "\n")
board_p = Path("${BOARD}")
if board_p.is_file():
    b = json.loads(board_p.read_text())
    st = b.setdefault("methods", {}).setdefault("proto_retrieval", {}).setdefault("fdg_test20", {})
    st.update({
        "status": "done",
        "stamp": "${STAMP}",
        "training_free": True,
        "support_pool": "FDG70%",
        "mean": md,
        "fold_dice": {},
        "note": f"DONE · FDG70% → FDG20% · {md:.3f}",
        "metric": "Dice; retrieve FDG70% + prototype",
    })
    b["updated_note"] = f"Proto+ FDG70% TEST={md:.3f}"
    board_p.write_text(json.dumps(b, indent=2) + "\n")
print(json.dumps(agg, indent=2))
PY

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true
echo "[proto-fdg70] ALL DONE ${STAMP}"
