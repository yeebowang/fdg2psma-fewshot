#!/usr/bin/env bash
# DpDNet FDG · 3×GPU (0,1,3) · per-GPU bs=2 → plans global_bs=6 · n_proc_DA=6/卡
# Schedule defaults: 169ep tr70/val0 (legacy). MAE-fullcase overrides via env
# (100ep · full-case tr/val · val every20 · best=ema_fg_dice).
#   bash ICLR2026/run/run_dpdnet_fdg_3gpu_bs2_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${WORK}/01_train_vis"
ICLR_VIS="${CTRL}/ICLR2026/vis"
DPD="${CTRL}/ICLR2026/third_party/DpDNet"
IMAGE="${TASK1_NNUNET_IMAGE:-autopet_baseline:latest}"

DATASET_ID="${DATASET_ID:-239}"
DS="Dataset${DATASET_ID}_DpDNet_FDG_2ch"
TRAINER="${TRAINER:-STUNetTrainer_small_prompt}"
CONFIG="${CONFIG:-3d_fullres}"
FOLD="${FOLD:-0}"
GPU_CSV="${TASK1_DPDNET_GPUS:-0,1,3}"
N_GPU="$(awk -F',' '{print NF}' <<<"${GPU_CSV}")"
BS_PER_GPU="${TASK1_DPDNET_BATCH_SIZE_PER_GPU:-2}"
GLOBAL_BS=$((BS_PER_GPU * N_GPU))
TOTAL_EPOCHS="${TASK1_DPDNET_NUM_EPOCHS:-169}"
TRAIN_ITERS="${TASK1_DPDNET_TRAIN_ITERS:-${TASK1_TRAIN_ITERS_PER_EPOCH:-70}}"
VAL_ITERS="${TASK1_DPDNET_VAL_ITERS:-${TASK1_VAL_ITERS_PER_EPOCH:-0}}"
VAL_EVERY="${TASK1_DPDNET_VAL_EVERY:-${TASK1_VAL_EVERY_N_EPOCHS:-1}}"
BEST_BY="${TASK1_BEST_BY:-}"
# per-rank DA workers (nnUNet Baseline1: TASK1_N_PROC_DA=6 → debug num_processes=6/rank)
N_PROC_DA="${TASK1_DPDNET_N_PROC_DA:-6}"
PLANS="${WORK}/nnUNet_preprocessed/${DS}/nnUNetPlans.json"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"

mkdir -p "${VIS}" "${ICLR_VIS}"

# plans batch_size = GLOBAL (nnUNet DDP splits across ranks)
python3 - <<PY
import json
from pathlib import Path
p = Path("${PLANS}")
d = json.loads(p.read_text())
d.setdefault("configurations", {}).setdefault("3d_fullres", {})["batch_size"] = int("${GLOBAL_BS}")
p.write_text(json.dumps(d, indent=2) + "\n")
print("[dpdnet-3gpu] plans batch_size=${GLOBAL_BS} (=${BS_PER_GPU}/GPU × ${N_GPU})")
PY

if [[ "${TASK1_DPDNET_SKIP_PREPARE:-0}" != "1" ]]; then
  docker run --rm --user root \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    --entrypoint python3 "${IMAGE}" \
    "${CTRL}/ICLR2026/scripts/prepare_dpdnet_fdg_dataset239.py" \
      --work "${WORK}" \
      --dst-id "${DATASET_ID}" \
      --batch-size "${GLOBAL_BS}"
  docker run --rm --user root -v "${DATA}:${DATA}" --entrypoint bash "${IMAGE}" -lc \
    "chown -R $(id -u):$(id -g) '${WORK}/nnUNet_preprocessed/${DS}' '${WORK}/nnUNet_raw/${DS}' 2>/dev/null || true"
fi

STAMP_TZ="${TASK1_STAMP_TZ:-Asia/Shanghai}"
GPU_TAG="${GPU_CSV//,/}"
if [[ -n "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME}"
else
  if [[ "${VAL_EVERY}" -gt 1 ]]; then
    STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_dpdnet_fdg_${N_GPU}gpu_bs${BS_PER_GPU}_gbs${GLOBAL_BS}_n${N_PROC_DA}_tr${TRAIN_ITERS}_val${VAL_ITERS}e${VAL_EVERY}_${TOTAL_EPOCHS}ep_gpu${GPU_TAG}"
  else
    STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_dpdnet_fdg_${N_GPU}gpu_bs${BS_PER_GPU}_gbs${GLOBAL_BS}_n${N_PROC_DA}_tr${TRAIN_ITERS}_val${VAL_ITERS}_${TOTAL_EPOCHS}ep_gpu${GPU_TAG}"
  fi
fi
export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_BASE="${DATA}"

RESULTS_ROOT="${WORK}/nnUNet_results/${STAMP}"
mkdir -p "${RESULTS_ROOT}"
docker run --rm --user root -v "${DATA}:${DATA}" --entrypoint bash "${IMAGE}" -lc \
  "chmod -R a+rwX '${RESULTS_ROOT}' 2>/dev/null || true" || true

LOG="${ICLR_VIS}/nohup_dpdnet_fdg_${STAMP}.log"
CNAME="dpdnet_fdg_${STAMP}"
INNER_CVD="$(seq -s, 0 $((N_GPU - 1)))"

echo "[dpdnet-fdg] STAMP=${STAMP} gpus=${GPU_CSV} n_gpu=${N_GPU} bs/gpu=${BS_PER_GPU} gbs=${GLOBAL_BS} ep=${TOTAL_EPOCHS} tr=${TRAIN_ITERS} val=${VAL_ITERS}e${VAL_EVERY} best=${BEST_BY:-ema_fg_dice} n_proc_DA=${N_PROC_DA}/rank"

export TASK1_PREFLIGHT_GPUS="${GPU_CSV}"
export TASK1_PREFLIGHT_LABEL="iclr2026-dpdnet-fdg-3gpu"
bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"dpdnet\":{\"fdg_pretrain\":{\"status\":\"running\",\"stamp\":\"${STAMP}\",\"bs\":${BS_PER_GPU},\"gbs\":${GLOBAL_BS},\"bs_note\":\"gbs=${GLOBAL_BS}\",\"total_epochs\":${TOTAL_EPOCHS},\"train_iters\":${TRAIN_ITERS},\"val_iters\":${VAL_ITERS},\"n_proc_da\":${N_PROC_DA},\"note\":\"STUNet · ${N_GPU}gpu gbs=${GLOBAL_BS} · tr${TRAIN_ITERS}/val${VAL_ITERS}e${VAL_EVERY} · best=${BEST_BY:-ema_fg_dice}\"}}},\"updated_note\":\"DpDNet FDG ${N_GPU}gpu gbs=${GLOBAL_BS} tr${TRAIN_ITERS}/val${VAL_ITERS}e${VAL_EVERY} ${TOTAL_EPOCHS}ep\"}" || true

docker rm -f "${CNAME}" >/dev/null 2>&1 || true
rm -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${STAMP}.txt"

# Raise NCCL timeout: rank0 unpack/verify of large Dataset239 can exceed default 10min
nohup docker run --rm \
  --name "${CNAME}" \
  --gpus "\"device=${GPU_CSV}\"" \
  -e CUDA_VISIBLE_DEVICES="${INNER_CVD}" \
  -e HOME=/home/algorithm \
  -e TASK1_DPDNET_NUM_EPOCHS="${TOTAL_EPOCHS}" \
  -e TASK1_DPDNET_TRAIN_ITERS="${TRAIN_ITERS}" \
  -e TASK1_DPDNET_VAL_ITERS="${VAL_ITERS}" \
  -e TASK1_DPDNET_VAL_EVERY="${VAL_EVERY}" \
  -e TASK1_TRAIN_ITERS_PER_EPOCH="${TRAIN_ITERS}" \
  -e TASK1_VAL_ITERS_PER_EPOCH="${VAL_ITERS}" \
  -e TASK1_VAL_EVERY_N_EPOCHS="${VAL_EVERY}" \
  -e TASK1_BEST_BY="${BEST_BY}" \
  -e nnUNet_raw="${WORK}/nnUNet_raw" \
  -e nnUNet_preprocessed="${WORK}/nnUNet_preprocessed" \
  -e nnUNet_results="${RESULTS_ROOT}" \
  -e PYTHONPATH="${DPD}:/home/algorithm/.local/lib/python3.11/site-packages" \
  -e nnUNet_n_proc_DA="${N_PROC_DA}" \
  -e NCCL_TIMEOUT=1800 \
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "${CTRL}:${CTRL}" \
  -v "${DATA}:${DATA}" \
  --shm-size=32g \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "mkdir -p '${RESULTS_ROOT}' && python3 -c 'import nnunetv2,nnunetv2.training.nnUNetTrainer.STUNetTrainer as T; print(\"nnunet\", nnunetv2.__file__); print(\"stunet\", T.__file__)' && nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} -tr ${TRAINER} -num_gpus ${N_GPU}" \
  >"${LOG}" 2>&1 &
echo $! > "${RESULTS_ROOT}/nohup.pid"
sleep 10

export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_GUARD_STAMP="${STAMP}"
export TASK1_GUARD_TRAINER_FOLDER="${TRAINER}__nnUNetPlans__${CONFIG}"
export TASK1_GUARD_DATASET_DIR="${DS}"
export TASK1_GUARD_TOTAL_EPOCHS="${TOTAL_EPOCHS}"
export TASK1_GUARD_RESTART_SCRIPT="ICLR2026/run/run_dpdnet_fdg_3gpu_bs2_bg.sh"
export TASK1_GUARD_REQUIRE_ARM=1
export FOLD="${FOLD}"
export TASK1_GUARD_EXTRA_ENV="TASK1_DPDNET_SKIP_PREPARE=1,TASK1_DPDNET_BATCH_SIZE_PER_GPU=${BS_PER_GPU},TASK1_DPDNET_GPUS=${GPU_CSV},TASK1_DPDNET_NUM_EPOCHS=${TOTAL_EPOCHS},TASK1_DPDNET_TRAIN_ITERS=${TRAIN_ITERS},TASK1_DPDNET_VAL_ITERS=${VAL_ITERS},TASK1_DPDNET_VAL_EVERY=${VAL_EVERY},TASK1_VAL_EVERY_N_EPOCHS=${VAL_EVERY},TASK1_BEST_BY=${BEST_BY},TASK1_DPDNET_N_PROC_DA=${N_PROC_DA},TASK1_NNUNET_RESULTS_STAMP_NAME=${STAMP}"
bash "${CTRL}/run_task/run_task1_train_auto_resume_guard_bg.sh" || true
# long FDG stage: arm so mid-train crash can CONTINUE (same as baseline1 FDG)
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true

echo "[dpdnet-fdg] launched pid=$(cat "${RESULTS_ROOT}/nohup.pid") log=${LOG}"
{
  echo "STAMP=${STAMP}"
  echo "LOG=${LOG}"
  echo "GPUS=${GPU_CSV} BS_PER_GPU=${BS_PER_GPU} GBS=${GLOBAL_BS} N_PROC_DA=${N_PROC_DA} EP=${TOTAL_EPOCHS} TR=${TRAIN_ITERS} VAL=${VAL_ITERS}e${VAL_EVERY} BEST=${BEST_BY:-ema_fg_dice}"
} > "${ICLR_VIS}/iclr2026_dpdnet_fdg_${STAMP}.txt"
echo "${STAMP}" > "${ICLR_VIS}/dpdnet_fdg_LAST_STAMP.txt"
