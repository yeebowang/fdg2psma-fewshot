#!/usr/bin/env bash
# SegAnyPET PSMA TEST20 click Dice · 1 container/fold · no shard
#   STAMP=<fewshot_stamp> [TASK1_FOLD_GPUS=2:0,5:1,8:3] bash ICLR2026/run/run_eval_seganypet_psma_test20_f258_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"
SEG_CODE="${CTRL}/ICLR2026/third_party/SegAnyPET/code"
SEG_PIP="${CTRL}/ICLR2026/third_party/seganypet_pip"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"
BOARD_STAGE="${TASK1_PSMA_BOARD_STAGE:-psma_fs${TASK1_FEWSHOT_N:-50}_f258}"

STAMP="${STAMP:?set STAMP=fewshot run stamp}"
FOLDS_CSV="${TASK1_SEGANY_FOLDS_CSV:-2,5,8}"
FOLD_GPUS_CSV="${TASK1_FOLD_GPUS:-2:0,5:1,8:3}"
TEST_ROOT="${TASK1_SEGANY_TEST20_ROOT:-${DATA}/task1_train_workspace/seganypet_psma_test20}"
CLICKS="${TASK1_SEGANY_TEST_CLICKS:-5}"
SKIP_DONE="${TASK1_TEST_SKIP_DONE:-1}"

OUT_ROOT="${REPO}/runs/${STAMP}"
EVAL_ROOT="${OUT_ROOT}/psma_test20_eval"
mkdir -p "${EVAL_ROOT}" "${LOG_DIR}"

declare -A GPU_OF
IFS=',' read -r -a _pairs <<< "${FOLD_GPUS_CSV}"
for p in "${_pairs[@]}"; do
  GPU_OF["${p%%:*}"]="${p##*:}"
done
IFS=',' read -r -a FOLD_ARR <<< "${FOLDS_CSV}"

echo "[seganypet-test20] STAMP=${STAMP} map=${FOLD_GPUS_CSV} clicks=${CLICKS}"

python3 "${CTRL}/ICLR2026/scripts/prepare_seganypet_psma_test20.py" \
  --out-root "${TEST_ROOT}"

_run_fold() {
  local fold="$1" gpu="$2"
  local out_dir="${OUT_ROOT}/seganypet/fold${fold}"
  local ckpt="${out_dir}/best.pth"
  [[ -f "${ckpt}" ]] || ckpt="${out_dir}/latest.pth"
  [[ -f "${ckpt}" ]] || { echo "[error] missing ckpt fold${fold}" >&2; return 1; }
  local out_json="${EVAL_ROOT}/fold${fold}_test20.json"
  local log="${LOG_DIR}/nohup_seganypet_test20_fold${fold}_${STAMP}.log"
  local cname="seganypet_test20_f${fold}_${STAMP}"

  if [[ "${SKIP_DONE}" == "1" && -f "${out_json}" ]]; then
    echo "[seganypet-test20] skip fold${fold}"
    return 0
  fi

  echo "[seganypet-test20] fold${fold} → GPU${gpu}"
  docker rm -f "${cname}" >/dev/null 2>&1 || true
  docker run --rm \
    --name "${cname}" \
    --gpus "device=${gpu}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONPATH="${SEG_PIP}:${SEG_CODE}:${CTRL}/ICLR2026/scripts" \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    -w "${SEG_CODE}" --shm-size=8g \
    "${IMAGE}" \
    python3 "${CTRL}/ICLR2026/scripts/seganypet_eval_psma_test20_fold.py" \
      --ckpt "${ckpt}" \
      --test-root "${TEST_ROOT}" \
      --pred-dir "${EVAL_ROOT}/fold${fold}_pred" \
      --out-json "${out_json}" \
      --fold "${fold}" \
      --stamp "${STAMP}" \
      --num-clicks "${CLICKS}" \
    >"${log}" 2>&1
}

pids=()
for fold in "${FOLD_ARR[@]}"; do
  gpu="${GPU_OF[${fold}]:-}"
  [[ -n "${gpu}" ]] || { echo "[error] no GPU for fold ${fold}" >&2; exit 1; }
  _run_fold "${fold}" "${gpu}" &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do
  wait "${pid}" || rc=1
done
[[ "${rc}" -eq 0 ]] || { echo "[error] some TEST folds failed" >&2; exit 1; }

# CPU FP/FN: host python has no nibabel — always score in docker, all folds 0–8.
# Never write mean_fp/mean_fn=null (that wiped the 3-fold TEST20 rates).
python3 "${CTRL}/ICLR2026/scripts/seganypet_recompute_test20_fpfn.py" \
  --stamp "${STAMP}" || true
echo "[seganypet-test20] DONE ${STAMP}"
