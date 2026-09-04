#!/usr/bin/env bash
# ICLR2026 · MAE SwinUNETR-Base FDG finetune
# - splits: baseline1 FDG 70% train / 10% val (711 / 101)
# - global bs=6 = 2/GPU × GPU 0,1,3
# - epochs=100 (上游库下游默认)
# - load: weights/swinv2base/swin_mae_best_v2.pth
# - 后程 20 ep：FDG+PSMA 双路 val → loss 3 曲线 (train / val_loss / PSMA_val_loss)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"

TOTAL_EPOCHS="${TASK1_MAE_NUM_EPOCHS:-100}"
BATCH_SIZE="${TASK1_MAE_BATCH_SIZE:-6}"  # global 2*3
GPUS="${TASK1_CUDA_VISIBLE_DEVICES:-0,1,3}"
DOCKER_GPUS="${TASK1_DOCKER_GPUS:-device=${GPUS}}"
PREFLIGHT_GPUS="${TASK1_PREFLIGHT_GPUS:-0 1 3}"
WORKERS_PREP="${TASK1_MAE_PREP_WORKERS:-16}"
WORKERS_TRAIN="${TASK1_MAE_TRAIN_WORKERS:-8}"
VAL_INTERVAL="${TASK1_MAE_VAL_INTERVAL:-20}"
FRESH="${TASK1_MAE_FRESH:-0}"
LATE_DUAL_EPOCHS="${TASK1_MAE_LATE_DUAL_EPOCHS:-20}"
PSMA_VAL_FROM="${TASK1_MAE_PSMA_VAL_FROM_EPOCH:-0}"  # 0 => last LATE_DUAL_EPOCHS

SPLITS_JSON="${TASK1_SPLITS_FINAL_JSON:-${CTRL}/ICLR2026/data/splits_baseline1_fdg_nnunet.json}"
PSMA_VAL_JSON="${TASK1_PSMA_VAL_CASES_JSON:-${CTRL}/ICLR2026/data/splits_baseline1_psma_val.json}"
CACHE_DIR="${TASK1_MAE_CACHE_DIR:-${DATA}/task1_train_workspace/mae_cache/fdg_baseline1_70_10}"
PSMA_CACHE_DIR="${TASK1_MAE_PSMA_CACHE_DIR:-${CACHE_DIR}}"
FOUNDATION="${TASK1_MAE_FOUNDATION:-${REPO}/weights/swinv2base/swin_mae_best_v2.pth}"
FOUNDATION_KIND="${TASK1_MAE_FOUNDATION_KIND:-mae}"

STAMP_TZ="${TASK1_STAMP_TZ:-Asia/Shanghai}"
if [[ -n "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME}"
elif [[ "${FOUNDATION_KIND}" == "none" ]]; then
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_mae_scratch_fdg_swinbase_gpu013_bs${BATCH_SIZE}_tr70_val10_${TOTAL_EPOCHS}ep"
else
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_mae_fdg_swinbase_gpu013_bs${BATCH_SIZE}_tr70_val10_${TOTAL_EPOCHS}ep"
fi
OUT_DIR="${TASK1_MAE_OUT_DIR:-${REPO}/runs/${STAMP}}"
LOG_DIR="${CTRL}/ICLR2026/vis"
LOSS_PNG="${TASK1_LOSS_OUT_NAME:-${LOG_DIR}/loss_curve_iclr2026_mae_fdg_${STAMP}.png}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${CACHE_DIR}" "${PSMA_CACHE_DIR}"

echo "[mae-fdg] STAMP=${STAMP}"
echo "[mae-fdg] foundation=${FOUNDATION} kind=${FOUNDATION_KIND}"
echo "[mae-fdg] splits=${SPLITS_JSON}"
echo "[mae-fdg] psma_val=${PSMA_VAL_JSON}"
echo "[mae-fdg] cache=${CACHE_DIR}"
echo "[mae-fdg] out=${OUT_DIR}"
echo "[mae-fdg] gpus=${GPUS} global_bs=${BATCH_SIZE} epochs=${TOTAL_EPOCHS} late_dual=${LATE_DUAL_EPOCHS}"
echo "[mae-fdg] loss_png=${LOSS_PNG}"

if [[ "${FOUNDATION_KIND}" != "none" ]]; then
  [[ -f "${FOUNDATION}" ]] || { echo "[error] missing foundation: ${FOUNDATION}" >&2; exit 1; }
fi
[[ -f "${SPLITS_JSON}" ]] || { echo "[error] missing splits: ${SPLITS_JSON}" >&2; exit 1; }
[[ -f "${PSMA_VAL_JSON}" ]] || { echo "[error] missing PSMA val: ${PSMA_VAL_JSON}" >&2; exit 1; }

# GPU preflight (0/1/3)
export TASK1_PREFLIGHT_GPUS="${PREFLIGHT_GPUS}"
export TASK1_PREFLIGHT_LABEL="iclr2026-mae-fdg"
bash "${CTRL}/scripts/task1_gpu_train_preflight.sh"

# crash-monitor arm window (non-nnUNet run; still follow arm discipline)
export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_BASE="${DATA}"
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true

docker_run() {
  # shellcheck disable=SC2086
  docker run --rm \
    --gpus "\"${DOCKER_GPUS}\"" \
    -e CUDA_VISIBLE_DEVICES=0,1,2 \
    -v "${CTRL}:${CTRL}" \
    -v "${DATA}:${DATA}" \
    -w "${REPO}" \
    --shm-size=16g \
    "${IMAGE}" \
    "$@"
}

echo "[mae-fdg] preprocess FDG cache …"
docker_run python3 "${CTRL}/ICLR2026/scripts/mae_preprocess_fdg_cache.py" \
  --splits-json "${SPLITS_JSON}" \
  --images-tr "${DATA}/dataset1/imagesTr" \
  --labels-tr "${DATA}/dataset1/labelsTr" \
  --out-dir "${CACHE_DIR}" \
  --workers "${WORKERS_PREP}"

echo "[mae-fdg] preprocess PSMA val cache …"
docker_run python3 "${CTRL}/ICLR2026/scripts/mae_preprocess_fdg_cache.py" \
  --cases-json "${PSMA_VAL_JSON}" \
  --images-tr "${DATA}/dataset1/imagesTr" \
  --labels-tr "${DATA}/dataset1/labelsTr" \
  --out-dir "${PSMA_CACHE_DIR}" \
  --workers "${WORKERS_PREP}"

FRESH_FLAG=()
[[ "${FRESH}" == "1" ]] && FRESH_FLAG=(--fresh)

PSMA_FROM_ARGS=()
if [[ "${PSMA_VAL_FROM}" != "0" ]]; then
  PSMA_FROM_ARGS=(--psma-val-from-epoch "${PSMA_VAL_FROM}")
fi

NOHUP_LOG="${LOG_DIR}/nohup_mae_fdg_${STAMP}.log"
echo "[mae-fdg] launching train -> ${NOHUP_LOG}"

# disarm before starting next stage process (train after prep)
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

nohup docker run --rm \
  --name "mae_fdg_${STAMP}" \
  --gpus "\"${DOCKER_GPUS}\"" \
  -e CUDA_VISIBLE_DEVICES=0,1,2 \
  -v "${CTRL}:${CTRL}" \
  -v "${DATA}:${DATA}" \
  -w "${REPO}" \
  --shm-size=16g \
  "${IMAGE}" \
  python3 "${CTRL}/ICLR2026/scripts/mae_finetune_fdg_swinbase.py" \
    --cache-dir "${CACHE_DIR}" \
    --splits-json "${SPLITS_JSON}" \
    --foundation-ckpt "${FOUNDATION}" \
    --foundation-kind "${FOUNDATION_KIND}" \
    --out-dir "${OUT_DIR}" \
    --epochs "${TOTAL_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --sw-batch-size 2 \
    --val-interval "${VAL_INTERVAL}" \
    --num-workers "${WORKERS_TRAIN}" \
    --psma-val-json "${PSMA_VAL_JSON}" \
    --psma-cache-dir "${PSMA_CACHE_DIR}" \
    --late-dual-epochs "${LATE_DUAL_EPOCHS}" \
    --loss-png "${LOSS_PNG}" \
    "${PSMA_FROM_ARGS[@]}" \
    "${FRESH_FLAG[@]}" \
  >"${NOHUP_LOG}" 2>&1 &

echo $! > "${OUT_DIR}/nohup.pid"
echo "[mae-fdg] nohup_pid=$(cat "${OUT_DIR}/nohup.pid") container=mae_fdg_${STAMP}"
echo "[mae-fdg] logs: docker logs -f mae_fdg_${STAMP}  OR  tail -f ${NOHUP_LOG}"
echo "[mae-fdg] stop: docker stop mae_fdg_${STAMP}"
echo "STAMP=${STAMP}" > "${LOG_DIR}/iclr2026_mae_fdg_${STAMP}.txt"
echo "OUT_DIR=${OUT_DIR}" >> "${LOG_DIR}/iclr2026_mae_fdg_${STAMP}.txt"
echo "FOUNDATION=${FOUNDATION}" >> "${LOG_DIR}/iclr2026_mae_fdg_${STAMP}.txt"
echo "FOUNDATION_KIND=${FOUNDATION_KIND}" >> "${LOG_DIR}/iclr2026_mae_fdg_${STAMP}.txt"
echo "batch_size=${BATCH_SIZE} epochs=${TOTAL_EPOCHS} gpus=${GPUS} late_dual=${LATE_DUAL_EPOCHS}" >> "${LOG_DIR}/iclr2026_mae_fdg_${STAMP}.txt"
echo "loss_png=${LOSS_PNG}" >> "${LOG_DIR}/iclr2026_mae_fdg_${STAMP}.txt"

# arm after train container is up
sleep 3
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true
