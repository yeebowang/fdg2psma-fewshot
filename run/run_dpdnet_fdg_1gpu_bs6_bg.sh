#!/usr/bin/env bash
# DpDNet FDG pretrain · Dataset239 · STUNetTrainer_small_prompt · 1×GPU bs=6
# Schedule: train_iters=70, val_iters=0, 1000ep
#   TASK1_DPDNET_GPU=0 bash ICLR2026/run/run_dpdnet_fdg_1gpu_bs6_bg.sh
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
GPU_ID="${TASK1_DPDNET_GPU:-0}"
BATCH_SIZE="${TASK1_DPDNET_BATCH_SIZE:-6}"
TOTAL_EPOCHS="${TASK1_DPDNET_NUM_EPOCHS:-169}"
TRAIN_ITERS="${TASK1_DPDNET_TRAIN_ITERS:-${TASK1_TRAIN_ITERS_PER_EPOCH:-70}}"
VAL_ITERS="${TASK1_DPDNET_VAL_ITERS:-${TASK1_VAL_ITERS_PER_EPOCH:-0}}"
N_PROC_DA="${TASK1_DPDNET_N_PROC_DA:-6}"

mkdir -p "${VIS}" "${ICLR_VIS}"

# 1) prepare dataset (hardlink + inject prompt type) — needs numpy (use docker)
if [[ "${TASK1_DPDNET_SKIP_PREPARE:-0}" != "1" ]]; then
  docker run --rm --user root \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    --entrypoint python3 "${IMAGE}" \
    "${CTRL}/ICLR2026/scripts/prepare_dpdnet_fdg_dataset239.py" \
      --work "${WORK}" \
      --dst-id "${DATASET_ID}" \
      --batch-size "${BATCH_SIZE}"
  docker run --rm --user root -v "${DATA}:${DATA}" --entrypoint bash "${IMAGE}" -lc \
    "chown -R $(id -u):$(id -g) '${WORK}/nnUNet_preprocessed/${DS}' '${WORK}/nnUNet_raw/${DS}' 2>/dev/null || true"
fi

STAMP_TZ="${TASK1_STAMP_TZ:-Asia/Shanghai}"
if [[ -n "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME}"
elif [[ "${TASK1_BOARD_METHOD:-}" == "dpdnet_dualenc" ]]; then
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_dpdnet_dualenc_fdg_1gpu_bs${BATCH_SIZE}_n${N_PROC_DA}_tr${TRAIN_ITERS}_val${VAL_ITERS}_${TOTAL_EPOCHS}ep_gpu${GPU_ID}"
else
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_dpdnet_fdg_1gpu_bs${BATCH_SIZE}_n${N_PROC_DA}_tr${TRAIN_ITERS}_val${VAL_ITERS}_${TOTAL_EPOCHS}ep_gpu${GPU_ID}"
fi
export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_BASE="${DATA}"

RESULTS_ROOT="${WORK}/nnUNet_results/${STAMP}"
mkdir -p "${RESULTS_ROOT}"
# container user `algorithm` must write into RESULTS_ROOT
docker run --rm --user root -v "${DATA}:${DATA}" --entrypoint bash "${IMAGE}" -lc \
  "chmod -R a+rwX '${RESULTS_ROOT}' 2>/dev/null || true" || true

LOG="${ICLR_VIS}/nohup_dpdnet_fdg_${STAMP}.log"
CNAME="dpdnet_fdg_${STAMP}"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"

BOARD_METHOD="${TASK1_BOARD_METHOD:-dpdnet}"
echo "[dpdnet-fdg] STAMP=${STAMP} ds=${DS} trainer=${TRAINER} gpu=${GPU_ID} bs=${BATCH_SIZE} ep=${TOTAL_EPOCHS} train_iters=${TRAIN_ITERS} val_iters=${VAL_ITERS} n_proc_DA=${N_PROC_DA} board=${BOARD_METHOD}"

export TASK1_PREFLIGHT_GPUS="${GPU_ID}"
export TASK1_PREFLIGHT_LABEL="iclr2026-dpdnet-fdg"
bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"fdg_pretrain\":{\"status\":\"running\",\"stamp\":\"${STAMP}\",\"bs\":${BATCH_SIZE},\"bs_note\":\"per-GPU 1card gpu${GPU_ID}\",\"total_epochs\":${TOTAL_EPOCHS},\"train_iters\":${TRAIN_ITERS},\"val_iters\":${VAL_ITERS},\"note\":\"${TRAINER} · tr${TRAIN_ITERS}/val${VAL_ITERS} · prompt=lymp(all FDG)\"}}},\"updated_note\":\"${BOARD_METHOD} FDG 1gpu bs=${BATCH_SIZE} tr${TRAIN_ITERS}/val${VAL_ITERS} ${TOTAL_EPOCHS}ep GPU${GPU_ID}\"}" || true

docker rm -f "${CNAME}" >/dev/null 2>&1 || true
rm -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${STAMP}.txt"

nohup docker run --rm \
  --name "${CNAME}" \
  --gpus "\"device=${GPU_ID}\"" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HOME=/home/algorithm \
  -e TASK1_DPDNET_NUM_EPOCHS="${TOTAL_EPOCHS}" \
  -e TASK1_DPDNET_TRAIN_ITERS="${TRAIN_ITERS}" \
  -e TASK1_DPDNET_VAL_ITERS="${VAL_ITERS}" \
  -e TASK1_TRAIN_ITERS_PER_EPOCH="${TRAIN_ITERS}" \
  -e TASK1_VAL_ITERS_PER_EPOCH="${VAL_ITERS}" \
  -e nnUNet_raw="${WORK}/nnUNet_raw" \
  -e nnUNet_preprocessed="${WORK}/nnUNet_preprocessed" \
  -e nnUNet_results="${RESULTS_ROOT}" \
  -e PYTHONPATH="${DPD}:/home/algorithm/.local/lib/python3.11/site-packages" \
  -e nnUNet_n_proc_DA="${N_PROC_DA}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TASK1_DPDNET_CT_ENCODER="${TASK1_DPDNET_CT_ENCODER:-}" \
  -e TASK1_DPDNET_PET_ENCODER="${TASK1_DPDNET_PET_ENCODER:-}" \
  -e TASK1_DPDNET_SKIP_ENCODER_INIT="${TASK1_DPDNET_SKIP_ENCODER_INIT:-0}" \
  -v "${CTRL}:${CTRL}" \
  -v "${DATA}:${DATA}" \
  --shm-size=16g \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "mkdir -p '${RESULTS_ROOT}' && python3 -c 'import nnunetv2,nnunetv2.training.nnUNetTrainer.STUNetTrainer as T; print(\"nnunet\", nnunetv2.__file__); print(\"stunet\", T.__file__)' && nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} -tr ${TRAINER}" \
  >"${LOG}" 2>&1 &
echo $! > "${RESULTS_ROOT}/nohup.pid"
sleep 8

# auto-resume guard (require_arm) then arm crash window
export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_GUARD_STAMP="${STAMP}"
export TASK1_GUARD_TRAINER_FOLDER="${TRAINER}__nnUNetPlans__${CONFIG}"
export TASK1_GUARD_DATASET_DIR="${DS}"
export TASK1_GUARD_TOTAL_EPOCHS="${TOTAL_EPOCHS}"
export TASK1_GUARD_RESTART_SCRIPT="ICLR2026/run/run_dpdnet_fdg_1gpu_bs6_bg.sh"
export TASK1_GUARD_REQUIRE_ARM=1
export FOLD="${FOLD}"
export TASK1_GUARD_EXTRA_ENV="TASK1_DPDNET_SKIP_PREPARE=1,TASK1_DPDNET_BATCH_SIZE=${BATCH_SIZE},TASK1_DPDNET_GPU=${GPU_ID},TASK1_DPDNET_NUM_EPOCHS=${TOTAL_EPOCHS},TASK1_DPDNET_TRAIN_ITERS=${TRAIN_ITERS},TASK1_DPDNET_VAL_ITERS=${VAL_ITERS},TASK1_DPDNET_N_PROC_DA=${N_PROC_DA},TASK1_NNUNET_RESULTS_STAMP_NAME=${STAMP},TRAINER=${TRAINER},TASK1_BOARD_METHOD=${BOARD_METHOD},TASK1_DPDNET_LAST_STAMP_FILE=${TASK1_DPDNET_LAST_STAMP_FILE:-${ICLR_VIS}/dpdnet_fdg_LAST_STAMP.txt},TASK1_DPDNET_CT_ENCODER=${TASK1_DPDNET_CT_ENCODER:-},TASK1_DPDNET_PET_ENCODER=${TASK1_DPDNET_PET_ENCODER:-},TASK1_DPDNET_SKIP_ENCODER_INIT=${TASK1_DPDNET_SKIP_ENCODER_INIT:-0}"
bash "${CTRL}/run_task/run_task1_train_auto_resume_guard_bg.sh" || true
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true

echo "[dpdnet-fdg] launched pid=$(cat "${RESULTS_ROOT}/nohup.pid") log=${LOG}"
echo "STAMP=${STAMP}" > "${ICLR_VIS}/iclr2026_dpdnet_fdg_${STAMP}.txt"
echo "LOG=${LOG}" >> "${ICLR_VIS}/iclr2026_dpdnet_fdg_${STAMP}.txt"
echo "GPU=${GPU_ID} BS=${BATCH_SIZE} TRAIN_ITERS=${TRAIN_ITERS} VAL_ITERS=${VAL_ITERS} EP=${TOTAL_EPOCHS}" >> "${ICLR_VIS}/iclr2026_dpdnet_fdg_${STAMP}.txt"
echo "${STAMP}" > "${TASK1_DPDNET_LAST_STAMP_FILE:-${ICLR_VIS}/dpdnet_fdg_LAST_STAMP.txt}"
