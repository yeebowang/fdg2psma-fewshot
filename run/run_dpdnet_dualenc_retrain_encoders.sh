#!/usr/bin/env bash
# Local dual-encoder retrain (public UniSeg epoch_94 weights are not published).
#   MODALITY=ct|pet GPU_ID=0 bash ICLR2026/run/run_dpdnet_dualenc_retrain_encoders.sh
# Saves:
#   ICLR2026/3D-MAE-PET-CT/weights/dpdnet/best_encoder_{ct,pet}_epoch_94.pth
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
ICLR_VIS="${CTRL}/ICLR2026/vis"
ENC_ROOT="${CTRL}/ICLR2026/3D-MAE-PET-CT/weights/dpdnet"
DPD="${CTRL}/ICLR2026/third_party/DpDNet"
IMAGE="${TASK1_NNUNET_IMAGE:-autopet_baseline:latest}"

MODALITY="${MODALITY:?need MODALITY=ct|pet}"
GPU_ID="${GPU_ID:-${TASK1_DPDNET_GPU:-0}}"
GPU_ID="${GPU_ID%%,*}"
EPOCHS="${TASK1_DPDNET_ENC_EPOCHS:-94}"
TRAIN_ITERS="${TASK1_DPDNET_ENC_TRAIN_ITERS:-70}"
VAL_ITERS="${TASK1_DPDNET_ENC_VAL_ITERS:-0}"
BATCH="${TASK1_DPDNET_ENC_BATCH:-6}"

if [[ "${MODALITY}" == "ct" ]]; then
  DATASET_ID=250
  DS="Dataset250_DpDNet_FDG_CT1ch"
  OUT_NAME="best_encoder_ct_epoch_94.pth"
elif [[ "${MODALITY}" == "pet" ]]; then
  DATASET_ID=251
  DS="Dataset251_DpDNet_FDG_PET1ch"
  OUT_NAME="best_encoder_pet_epoch_94.pth"
else
  echo "[error] MODALITY must be ct or pet" >&2
  exit 2
fi

PP="${WORK}/nnUNet_preprocessed/${DS}"
[[ -f "${PP}/nnUNetPlans.json" ]] || { echo "[error] missing preprocess ${PP} — run CPU prepare first" >&2; exit 1; }
python3 "${CTRL}/ICLR2026/scripts/adapt_nnunet_plans_for_dpdnet.py" "${PP}/nnUNetPlans.json"
docker run --rm --user root \
  -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
  --entrypoint python3 "${IMAGE}" \
  "${CTRL}/ICLR2026/scripts/inject_dpdnet_prompt_type.py" \
  "${PP}/nnUNetPlans_3d_fullres" || true
mkdir -p "${ENC_ROOT}" "${ICLR_VIS}"
chmod a+rwX "${ENC_ROOT}" 2>/dev/null || true

STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_dpdnet_enc_${MODALITY}_1gpu_bs${BATCH}_tr${TRAIN_ITERS}_${EPOCHS}ep_gpu${GPU_ID}"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
RESULTS_ROOT="${WORK}/nnUNet_results/${STAMP}"
mkdir -p "${RESULTS_ROOT}"
docker run --rm --user root -v "${DATA}:${DATA}" --entrypoint bash "${IMAGE}" -lc \
  "chmod -R a+rwX '${RESULTS_ROOT}' '${ENC_ROOT}' 2>/dev/null || true" || true

LOG="${ICLR_VIS}/nohup_dpdnet_enc_${MODALITY}_${STAMP}.log"
CNAME="dpdnet_enc_${MODALITY}_${STAMP}"
TRAINER="STUNetTrainer_small_prompt"

echo "[dualenc-retrain] ${MODALITY} gpu=${GPU_ID} ds=${DS} ep=${EPOCHS} → ${ENC_ROOT}/${OUT_NAME}"
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

docker rm -f "${CNAME}" >/dev/null 2>&1 || true
rm -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${STAMP}.txt"

nohup docker run --rm \
  --name "${CNAME}" \
  --gpus "\"device=${GPU_ID}\"" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HOME=/home/algorithm \
  -e TASK1_DPDNET_NUM_EPOCHS="${EPOCHS}" \
  -e TASK1_DPDNET_TRAIN_ITERS="${TRAIN_ITERS}" \
  -e TASK1_DPDNET_VAL_ITERS="${VAL_ITERS}" \
  -e TASK1_TRAIN_ITERS_PER_EPOCH="${TRAIN_ITERS}" \
  -e TASK1_VAL_ITERS_PER_EPOCH="${VAL_ITERS}" \
  -e nnUNet_raw="${WORK}/nnUNet_raw" \
  -e nnUNet_preprocessed="${WORK}/nnUNet_preprocessed" \
  -e nnUNet_results="${RESULTS_ROOT}" \
  -e PYTHONPATH="${DPD}:/home/algorithm/.local/lib/python3.11/site-packages" \
  -e nnUNet_n_proc_DA=6 \
  -e TASK1_DPDNET_SKIP_ENCODER_INIT=1 \
  -e TASK1_DPDNET_SKIP_FINAL_VAL=1 \
  -v "${CTRL}:${CTRL}" \
  -v "${DATA}:${DATA}" \
  --shm-size=16g \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "nnUNetv2_train ${DATASET_ID} 3d_fullres 0 -tr ${TRAINER}" \
  >"${LOG}" 2>&1 &
echo $! > "${RESULTS_ROOT}/nohup.pid"
echo "${STAMP}" > "${ICLR_VIS}/dpdnet_enc_${MODALITY}_LAST_STAMP.txt"

export TASK1_GUARD_STAMP="${STAMP}"
export TASK1_GUARD_DATASET_DIR="${DS}"
export TASK1_GUARD_TRAINER_FOLDER="${TRAINER}__nnUNetPlans__3d_fullres"
export TASK1_GUARD_TOTAL_EPOCHS="${EPOCHS}"
export TASK1_GUARD_RESTART_SCRIPT="ICLR2026/run/run_dpdnet_dualenc_retrain_encoders.sh"
export TASK1_GUARD_REQUIRE_ARM=1
export TASK1_GUARD_EXTRA_ENV="MODALITY=${MODALITY},GPU_ID=${GPU_ID},TASK1_DPDNET_ENC_EPOCHS=${EPOCHS},TASK1_NNUNET_RESULTS_STAMP_NAME=${STAMP}"
bash "${CTRL}/run_task/run_task1_train_auto_resume_guard_bg.sh" || true
# 阶段中 DISARM：guard 常驻但 require_arm，等 ckpt 写出后再 arm。
# 不要在刚启动训练时 arm，否则 600s 窗口会把初始化失败当成崩溃续训。

FOLD="${RESULTS_ROOT}/${DS}/${TRAINER}__nnUNetPlans__3d_fullres/fold_0"
echo "[dualenc-retrain] wait ${MODALITY} ${STAMP}"
while [[ ! -f "${FOLD}/checkpoint_final.pth" && ! -f "${FOLD}/checkpoint_best.pth" ]]; do
  if docker ps --format '{{.Names}}' | grep -qx "${CNAME}"; then
    sleep 90
    continue
  fi
  sleep 30
  break
done
CKPT=""
for f in checkpoint_final.pth checkpoint_best.pth checkpoint_latest.pth; do
  [[ -f "${FOLD}/${f}" ]] && { CKPT="${FOLD}/${f}"; break; }
done
[[ -n "${CKPT}" ]] || { echo "[error] no ${MODALITY} encoder ckpt" >&2; exit 1; }
TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true

docker run --rm \
  --entrypoint python3 \
  -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
  "${IMAGE}" \
  "${CTRL}/ICLR2026/scripts/extract_dpdnet_dualenc_encoder.py" \
    --ckpt "${CKPT}" --out "${ENC_ROOT}/${OUT_NAME}"
chmod a+r "${ENC_ROOT}/${OUT_NAME}" || true
echo "[dualenc-retrain] wrote ${ENC_ROOT}/${OUT_NAME}"
TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" TASK1_CRASH_MONITOR_STAGE=next_stage \
  bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
