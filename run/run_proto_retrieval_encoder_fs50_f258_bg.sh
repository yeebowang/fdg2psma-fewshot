#!/usr/bin/env bash
# Proto+Retrieval with fs50 retrieval-encoder fine-tune (SimCLR) + TEST20 f258 per fold.
#
#   bash ICLR2026/run/run_proto_retrieval_encoder_fs50_f258_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"

STAMP="${STAMP:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_proto_retrieval_encoder_fs50_f258_gpu013}"
GALLERY="${TASK1_PROTO_GALLERY:-${WORK}/proto_retrieval/fdg100_gallery.npz}"
STRAT="${CTRL}/ICLR2026/data/splits_stratified_70_10_20.json"
TEST_JSON="${CTRL}/ICLR2026/data/splits_mae_psma_test20.json"
IMG="${DATA}/dataset1/imagesTr"
LAB="${DATA}/dataset1/labelsTr"
OUT_ROOT="${CTRL}/ICLR2026/runs/proto_retrieval/${STAMP}"
ENC_ROOT="${OUT_ROOT}/encoder_fs50"
EVAL_ROOT="${OUT_ROOT}/psma_test20_eval"
TOPK="${TASK1_PROTO_TOPK:-3}"
POOL="${TASK1_PROTO_POOL:-fdg100_psma50}"
EPOCHS="${TASK1_PROTO_ENC_EPOCHS:-200}"
FOLDS="${TASK1_FOLDS:-2,5,8}"
FOLD_GPUS="${TASK1_FOLD_GPUS:-2:0,5:1,8:3}"
PIPE_LOG="${VIS}/nohup_proto_retrieval_encoder_${STAMP}.log"

exec > >(tee -a "${PIPE_LOG}") 2>&1
echo "[proto-enc] STAMP=${STAMP} pool=${POOL} topk=${TOPK} epochs=${EPOCHS}"

_dpy() {
  docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    -w "${CTRL}" \
    "${IMAGE}" \
    python3 "$@"
}

_dpy_gpu() {
  local gpu="$1"
  shift
  docker run --rm \
    --gpus "device=${gpu}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONUNBUFFERED=1 \
    --user "$(id -u):$(id -g)" \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    -w "${CTRL}" \
    "${IMAGE}" \
    python3 "$@"
}

if [[ ! -f "${GALLERY}" ]]; then
  echo "[proto-enc] building FDG100% gallery..."
  _dpy "${CTRL}/ICLR2026/scripts/proto_retrieval_build_gallery.py" \
    --stratified-json "${STRAT}" \
    --img-dir "${IMG}" \
    --lab-dir "${LAB}" \
    --out "${GALLERY}" \
    --workers "${TASK1_PROTO_GALLERY_WORKERS:-8}"
fi

mkdir -p "${ENC_ROOT}" "${EVAL_ROOT}"

declare -A GPU_OF
IFS=',' read -r -a _pairs <<< "${FOLD_GPUS}"
for p in "${_pairs[@]}"; do GPU_OF["${p%%:*}"]="${p##*:}"; done
IFS=',' read -r -a FOLD_ARR <<< "${FOLDS}"

_train_fold() {
  local fold="$1" gpu="$2"
  local ckpt="${ENC_ROOT}/encoder_fold${fold}.pt"
  local log="${VIS}/nohup_proto_enc_train_f${fold}_${STAMP}.log"
  echo "[proto-enc] train fold${fold} → GPU${gpu}"
  _dpy_gpu "${gpu}" \
    "${CTRL}/ICLR2026/scripts/proto_retrieval_train_encoder_fs50.py" \
    --fold "${fold}" \
    --img-dir "${IMG}" \
    --out "${ckpt}" \
    --epochs "${EPOCHS}" \
    >"${log}" 2>&1
}

_eval_fold() {
  local fold="$1"
  local ckpt="${ENC_ROOT}/encoder_fold${fold}.pt"
  local pred_dir="${EVAL_ROOT}/fold${fold}/predict"
  local out_json="${EVAL_ROOT}/fold${fold}_test20.json"
  local log="${VIS}/nohup_proto_enc_test20_f${fold}_${STAMP}.log"
  echo "[proto-enc] eval fold${fold} pool=${POOL}"
  _dpy \
    "${CTRL}/ICLR2026/scripts/proto_retrieval_eval_test20.py" \
    --gallery "${GALLERY}" \
    --cases-json "${TEST_JSON}" \
    --img-dir "${IMG}" \
    --lab-dir "${LAB}" \
    --pred-dir "${pred_dir}" \
    --out-json "${out_json}" \
    --fold "${fold}" \
    --topk "${TOPK}" \
    --pool-mode "${POOL}" \
    --encoder-ckpt "${ckpt}" \
    --stamp "${STAMP}" \
    --tag "proto_retrieval_enc_f${fold}" \
    >"${log}" 2>&1
}

pids=()
for fold in "${FOLD_ARR[@]}"; do
  gpu="${GPU_OF[${fold}]:-0}"
  _train_fold "${fold}" "${gpu}" &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
[[ "${rc}" -eq 0 ]] || { echo "[error] encoder train failed" >&2; exit 1; }

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
    fep[str(f)] = int(${EPOCHS})
vals = list(fd.values())
summary = {
    "stamp": stamp,
    "method": "proto_retrieval_encoder",
    "split": "PSMA_TEST20",
    "support_pool": "${POOL}",
    "encoder_epochs": int(${EPOCHS}),
    "topk": int("${TOPK}"),
    "fold_test_dice": fd,
    "fold_ckpt_ep": fep,
    "test_mean": sum(vals)/len(vals),
    "test_std": statistics.pstdev(vals) if len(vals)>1 else 0.0,
    "metric": "TEST20 Dice; fs50 SimCLR encoder + retrieve + prototype",
}
(eval_root / "aggregate_test20_f258.json").write_text(json.dumps(summary, indent=2)+"\n")
(vis / f"aggregate_proto_retrieval_encoder_fs50_f258_{stamp}.json").write_text(json.dumps(summary, indent=2)+"\n")
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
        "note": f"encoder fs50 {'${EPOCHS}'}ep · pool=${POOL} · TEST20 3/3",
        "support_pool": "${POOL}",
        "encoder_epochs": int(${EPOCHS}),
        "training_free": False,
        "topk": int("${TOPK}"),
    })
    b["updated_note"] = f"proto_retrieval+encoder TEST20 mean={summary['test_mean']:.3f}"
    board_p.write_text(json.dumps(b, indent=2)+"\n")
print(json.dumps(summary, indent=2))
PY

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" --plot-only || true
echo "[proto-enc] DONE ${STAMP} log=${PIPE_LOG}"
