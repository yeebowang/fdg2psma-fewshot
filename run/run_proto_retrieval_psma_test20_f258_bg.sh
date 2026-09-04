#!/usr/bin/env bash
# Prototype + Retrieval PSMA TEST20 f258
# Support pool: FDG 100% (1014 cases from stratified split); top-K retrieve + ALPNet-lite prototype.
#
#   bash ICLR2026/run/run_proto_retrieval_psma_test20_f258_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"

_dpy() {
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    -w "${CTRL}" \
    "${IMAGE}" \
    python3 "$@"
}

STAMP="${STAMP:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_proto_retrieval_fdg100_f258_gpu013}"
GALLERY="${TASK1_PROTO_GALLERY:-${WORK}/proto_retrieval/fdg100_gallery.npz}"
STRAT="${CTRL}/ICLR2026/data/splits_stratified_70_10_20.json"
TEST_JSON="${CTRL}/ICLR2026/data/splits_mae_psma_test20.json"
IMG="${DATA}/dataset1/imagesTr"
LAB="${DATA}/dataset1/labelsTr"
OUT_ROOT="${CTRL}/ICLR2026/runs/proto_retrieval/${STAMP}"
EVAL_ROOT="${OUT_ROOT}/psma_test20_eval"
TOPK="${TASK1_PROTO_TOPK:-3}"
FOLDS="${TASK1_FOLDS:-2,5,8}"
PIPE_LOG="${VIS}/nohup_proto_retrieval_test20_${STAMP}.log"

exec > >(tee -a "${PIPE_LOG}") 2>&1
echo "[proto-retrieval] STAMP=${STAMP} topk=${TOPK} gallery=${GALLERY}"

if [[ ! -f "${GALLERY}" ]]; then
  echo "[proto-retrieval] building FDG100% gallery..."
  _dpy "${CTRL}/ICLR2026/scripts/proto_retrieval_build_gallery.py" \
    --stratified-json "${STRAT}" \
    --img-dir "${IMG}" \
    --lab-dir "${LAB}" \
    --out "${GALLERY}" \
    --workers "${TASK1_PROTO_GALLERY_WORKERS:-8}"
fi

mkdir -p "${EVAL_ROOT}"
PRED_DIR="${EVAL_ROOT}/predict"
OUT_JSON="${EVAL_ROOT}/test20_all.json"
echo "[proto-retrieval] TEST20 eval (training-free, FDG100% pool, topk=${TOPK})"
_dpy "${CTRL}/ICLR2026/scripts/proto_retrieval_eval_test20.py" \
  --gallery "${GALLERY}" \
  --cases-json "${TEST_JSON}" \
  --img-dir "${IMG}" \
  --lab-dir "${LAB}" \
  --pred-dir "${PRED_DIR}" \
  --out-json "${OUT_JSON}" \
  --fold 0 \
  --topk "${TOPK}" \
  --stamp "${STAMP}" \
  --tag "proto_retrieval_fdg100"

IFS=',' read -r -a FOLD_ARR <<< "${FOLDS}"
for fold in "${FOLD_ARR[@]}"; do
  mkdir -p "${EVAL_ROOT}/fold${fold}"
  cp -a "${OUT_JSON}" "${EVAL_ROOT}/fold${fold}_test20.json"
  mkdir -p "${EVAL_ROOT}/fold${fold}/predict"
  cp -a "${PRED_DIR}/." "${EVAL_ROOT}/fold${fold}/predict/" 2>/dev/null || true
done

python3 - <<PY
import json, statistics
from pathlib import Path
eval_root = Path("${EVAL_ROOT}")
vis = Path("${VIS}")
stamp = "${STAMP}"
folds = [int(x) for x in "${FOLDS}".split(",")]
fd, fep = {}, {}
for f in folds:
    p = eval_root / f"fold{f}_test20.json"
    d = json.loads(p.read_text())
    fd[str(f)] = d["mean_dice"]
    fep[str(f)] = 0  # training-free; ckpt ep N/A
vals = list(fd.values())
summary = {
    "stamp": stamp,
    "method": "proto_retrieval",
    "split": "PSMA_TEST20",
    "support_pool": "FDG100%",
    "topk": int("${TOPK}"),
    "fold_test_dice": fd,
    "fold_ckpt_ep": fep,
    "test_mean": sum(vals)/len(vals),
    "test_std": statistics.pstdev(vals) if len(vals)>1 else 0.0,
    "metric": "TEST20 Dice; retrieve FDG100% + prototype",
}
(eval_root / "aggregate_test20_f258.json").write_text(json.dumps(summary, indent=2)+"\n")
(vis / f"aggregate_proto_retrieval_psma_test20_f258_{stamp}.json").write_text(json.dumps(summary, indent=2)+"\n")
board_p = Path("${BOARD}")
if board_p.is_file():
    b = json.loads(board_p.read_text())
    pr = b.setdefault("methods", {}).setdefault("proto_retrieval", {
        "label": "Proto+Retrieval (ECCV'20)",
        "fdg_pretrain": {
            "status": "n/a",
            "training_free": True,
            "support_pool": "FDG100%",
            "note": "FDG100% = supervised support gallery (not training)",
        },
        "psma_fs50_f258": {},
    })
    st = pr["psma_fs50_f258"]
    st.update({
        "status": "done",
        "stamp": stamp,
        "fold_dice": fd,
        "fold_ckpt_ep": fep,
        "mean": summary["test_mean"],
        "metric": summary["metric"],
        "note": "TEST20 DONE · 3/3",
        "support_pool": "FDG100%",
        "training_free": True,
        "topk": int("${TOPK}"),
    })
    b["updated_note"] = f"proto_retrieval TEST20 mean={summary['test_mean']:.3f}"
    board_p.write_text(json.dumps(b, indent=2)+"\n")
print(json.dumps(summary, indent=2))
PY

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" --plot-only || true
echo "[proto-retrieval] DONE ${STAMP} log=${PIPE_LOG}"
