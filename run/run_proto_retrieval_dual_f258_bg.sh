#!/usr/bin/env bash
# Dual-path Proto+Retrieval: PSMA fs50 top-2 + FDG100% top-2, PSMA vote weight=2.
#
#   bash ICLR2026/run/run_proto_retrieval_dual_f258_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"

STAMP="${STAMP:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_proto_retrieval_dual_f258_gpu013}"
GALLERY="${TASK1_PROTO_GALLERY:-${WORK}/proto_retrieval/fdg100_gallery.npz}"
TEST_JSON="${CTRL}/ICLR2026/data/splits_mae_psma_test20.json"
IMG="${DATA}/dataset1/imagesTr"
LAB="${DATA}/dataset1/labelsTr"
OUT_ROOT="${CTRL}/ICLR2026/runs/proto_retrieval/${STAMP}"
EVAL_ROOT="${OUT_ROOT}/psma_test20_eval"
PSMA_K="${TASK1_PROTO_PSMA_TOPK:-2}"
FDG_K="${TASK1_PROTO_FDG_TOPK:-2}"
PSMA_W="${TASK1_PROTO_PSMA_VOTE_WEIGHT:-2}"
FDG_W="${TASK1_PROTO_FDG_VOTE_WEIGHT:-1}"
FOLDS="${TASK1_FOLDS:-2,5,8}"
PIPE_LOG="${VIS}/nohup_proto_retrieval_dual_${STAMP}.log"

exec > >(tee -a "${PIPE_LOG}") 2>&1
echo "[proto-dual] STAMP=${STAMP} psma_topk=${PSMA_K} fdg_topk=${FDG_K} weights=${PSMA_W}:${FDG_W}"

_dpy() {
  docker run --rm \
    -e PYTHONUNBUFFERED=1 \
    --user "$(id -u):$(id -g)" \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    -w "${CTRL}" \
    "${IMAGE}" \
    python3 "$@"
}

[[ -f "${GALLERY}" ]] || { echo "[error] missing gallery ${GALLERY}" >&2; exit 1; }
mkdir -p "${EVAL_ROOT}"

_eval_fold() {
  local fold="$1"
  local pred_dir="${EVAL_ROOT}/fold${fold}/predict"
  local out_json="${EVAL_ROOT}/fold${fold}_test20.json"
  local log="${VIS}/nohup_proto_dual_test20_f${fold}_${STAMP}.log"
  echo "[proto-dual] eval fold${fold}"
  _dpy "${CTRL}/ICLR2026/scripts/proto_retrieval_eval_test20.py" \
    --gallery "${GALLERY}" \
    --cases-json "${TEST_JSON}" \
    --img-dir "${IMG}" \
    --lab-dir "${LAB}" \
    --pred-dir "${pred_dir}" \
    --out-json "${out_json}" \
    --fold "${fold}" \
    --pool-mode dual_psma_fdg \
    --psma-topk "${PSMA_K}" \
    --fdg-topk "${FDG_K}" \
    --psma-vote-weight "${PSMA_W}" \
    --fdg-vote-weight "${FDG_W}" \
    --stamp "${STAMP}" \
    --tag "proto_retrieval_dual_f${fold}" \
    >"${log}" 2>&1
}

IFS=',' read -r -a FOLD_ARR <<< "${FOLDS}"
pids=()
for fold in "${FOLD_ARR[@]}"; do
  _eval_fold "${fold}" &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
[[ "${rc}" -eq 0 ]] || { echo "[error] eval failed" >&2; exit 1; }

python3 - <<PY
import json, statistics
from pathlib import Path
eval_root = Path("${EVAL_ROOT}")
vis = Path("${VIS}")
stamp = "${STAMP}"
folds = [int(x) for x in "${FOLDS}".split(",")]
fd, fep = {}, {}
for f in folds:
    d = json.loads((eval_root / f"fold{f}_test20.json").read_text())
    fd[str(f)] = d["mean_dice"]
    fep[str(f)] = 0
vals = list(fd.values())
summary = {
    "stamp": stamp,
    "method": "proto_retrieval_dual",
    "split": "PSMA_TEST20",
    "support_pool": f"PSMAfs50 top${PSMA_K} + FDG100% top${FDG_K}",
    "vote_weights": {"psma": float("${PSMA_W}"), "fdg": float("${FDG_W}")},
    "fold_test_dice": fd,
    "fold_ckpt_ep": fep,
    "test_mean": sum(vals)/len(vals),
    "test_std": statistics.pstdev(vals) if len(vals)>1 else 0.0,
    "metric": "TEST20 Dice; dual retrieve + weighted prototype vote",
}
(eval_root / "aggregate_test20_f258.json").write_text(json.dumps(summary, indent=2)+"\n")
(vis / f"aggregate_proto_retrieval_dual_f258_{stamp}.json").write_text(json.dumps(summary, indent=2)+"\n")
board_p = Path("${BOARD}")
if board_p.is_file():
    b = json.loads(board_p.read_text())
    pr = b.setdefault("methods", {}).setdefault("proto_retrieval", {})
    st = pr.setdefault("psma_fs50_f258", {})
    st.update({
        "status": "done",
        "stamp": stamp,
        "fold_dice": fd,
        "fold_ckpt_ep": fep,
        "mean": summary["test_mean"],
        "metric": summary["metric"],
        "note": f"dual PSMA${PSMA_K}+FDG${FDG_K} w=${PSMA_W}:${FDG_W} · TEST20 3/3",
        "support_pool": summary["support_pool"],
        "training_free": True,
        "vote_weights": summary["vote_weights"],
    })
    b["updated_note"] = f"proto_retrieval dual TEST20 mean={summary['test_mean']:.3f}"
    board_p.write_text(json.dumps(b, indent=2)+"\n")
print(json.dumps(summary, indent=2))
PY

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" --plot-only || true
echo "[proto-dual] DONE ${STAMP} log=${PIPE_LOG}"
