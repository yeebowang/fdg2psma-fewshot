#!/usr/bin/env bash
# ICLR2026 · FDG MAE → continued SSL on unlabeled PSMA(+FDG align) → PSMA 10-shot finetune
#
# Stage A: continued MAE SSL
#   - init: weights/swinv2base/swin_mae_best_v2.pth (FDG MAE; released best, no local last)
#   - PSMA unlabeled ≤70% train (421) + FDG train for alignment
#   - tracer token + MMD feature alignment
# Stage B: few-shot finetune
#   - train: 10 labeled PSMA (stratified from 70% pool)
#   - val: PSMA 10% labeled (59)
#   - init: Stage A latest MAE
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"

SSL_EPOCHS="${TASK1_MAE_SSL_EPOCHS:-50}"
FT_EPOCHS="${TASK1_MAE_NUM_EPOCHS:-100}"
BATCH_SIZE="${TASK1_MAE_BATCH_SIZE:-6}"
SSL_BATCH_SIZE="${TASK1_MAE_SSL_BATCH_SIZE:-3}"  # must be divisible by n_gpu for DataParallel; dual MAE needs small bs
GPUS="${TASK1_CUDA_VISIBLE_DEVICES:-0,1,3}"
DOCKER_GPUS="${TASK1_DOCKER_GPUS:-device=${GPUS}}"
PREFLIGHT_GPUS="${TASK1_PREFLIGHT_GPUS:-0 1 3}"
WORKERS_PREP="${TASK1_MAE_PREP_WORKERS:-16}"
WORKERS_TRAIN="${TASK1_MAE_TRAIN_WORKERS:-8}"
ALIGN_W="${TASK1_MAE_ALIGN_WEIGHT:-0.1}"
LATE_DUAL_EPOCHS="${TASK1_MAE_LATE_DUAL_EPOCHS:-20}"
FRESH_SSL="${TASK1_MAE_SSL_FRESH:-1}"
FRESH_FT="${TASK1_MAE_FRESH:-1}"

PSMA_SPLITS="${CTRL}/ICLR2026/data/splits_baseline2_psma_uda_nnunet.json"
FDG_SPLITS="${CTRL}/ICLR2026/data/splits_baseline1_fdg_nnunet.json"
FEWSHOT_SPLITS="${CTRL}/ICLR2026/data/splits_mae_psma_fewshot10_nnunet.json"
PSMA_CACHE="${DATA}/task1_train_workspace/mae_cache/psma_baseline2_70_10"
FDG_CACHE="${DATA}/task1_train_workspace/mae_cache/fdg_baseline1_70_10"
FDG_MAE_CKPT="${TASK1_MAE_FDG_CKPT:-${REPO}/weights/swinv2base/swin_mae_best_v2.pth}"

STAMP_TZ="${TASK1_STAMP_TZ:-Asia/Shanghai}"
if [[ -n "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME}"
else
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_mae_psma_ssl_fewshot10_gpu013_bs${BATCH_SIZE}_ssl${SSL_EPOCHS}_ft${FT_EPOCHS}"
fi
OUT_SSL="${REPO}/runs/${STAMP}/ssl_continued"
OUT_FT="${REPO}/runs/${STAMP}/fewshot10"
LOG_DIR="${CTRL}/ICLR2026/vis"
mkdir -p "${OUT_SSL}" "${OUT_FT}" "${LOG_DIR}" "${PSMA_CACHE}" "${FDG_CACHE}"

SSL_PNG="${LOG_DIR}/loss_curve_iclr2026_mae_psma_ssl_${STAMP}.png"
FT_PNG="${LOG_DIR}/loss_curve_iclr2026_mae_psma_fewshot10_${STAMP}.png"
SSL_LATEST="${OUT_SSL}/swin_mae_psma_continued_latest.pth"

echo "[mae-ssl-fs] STAMP=${STAMP}"
echo "[mae-ssl-fs] fdg_mae_init=${FDG_MAE_CKPT}"
echo "[mae-ssl-fs] ssl_epochs=${SSL_EPOCHS} ft_epochs=${FT_EPOCHS} ssl_bs=${SSL_BATCH_SIZE} ft_bs=${BATCH_SIZE} align_w=${ALIGN_W}"

[[ -f "${FDG_MAE_CKPT}" ]] || { echo "[error] missing FDG MAE ckpt: ${FDG_MAE_CKPT}" >&2; exit 1; }

# fewshot splits
python3 "${CTRL}/ICLR2026/scripts/export_mae_psma_fewshot10.py" \
  --out-json "${FEWSHOT_SPLITS}" --n-shot 10 --seed 42

export TASK1_PREFLIGHT_GPUS="${PREFLIGHT_GPUS}"
export TASK1_PREFLIGHT_LABEL="iclr2026-mae-psma-ssl-fs"
bash "${CTRL}/scripts/task1_gpu_train_preflight.sh"

export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_BASE="${DATA}"

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

# ---------- Stage A: caches + SSL ----------
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true

echo "[mae-ssl-fs] preprocess PSMA 70% cache …"
docker_run python3 "${CTRL}/ICLR2026/scripts/mae_preprocess_fdg_cache.py" \
  --splits-json "${PSMA_SPLITS}" \
  --images-tr "${DATA}/dataset1/imagesTr" \
  --labels-tr "${DATA}/dataset1/labelsTr" \
  --out-dir "${PSMA_CACHE}" \
  --workers "${WORKERS_PREP}"

echo "[mae-ssl-fs] preprocess FDG train cache (align) …"
docker_run python3 "${CTRL}/ICLR2026/scripts/mae_preprocess_fdg_cache.py" \
  --splits-json "${FDG_SPLITS}" \
  --images-tr "${DATA}/dataset1/imagesTr" \
  --labels-tr "${DATA}/dataset1/labelsTr" \
  --out-dir "${FDG_CACHE}" \
  --workers "${WORKERS_PREP}"

bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

FRESH_SSL_FLAG=()
[[ "${FRESH_SSL}" == "1" ]] && FRESH_SSL_FLAG=(--fresh)

SSL_LOG="${LOG_DIR}/nohup_mae_psma_ssl_${STAMP}.log"
echo "[mae-ssl-fs] Stage A SSL -> ${SSL_LOG}"
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true

# run SSL in foreground docker (pipeline waits), log to file
set +e
docker run --rm \
  --name "mae_ssl_${STAMP}" \
  --gpus "\"${DOCKER_GPUS}\"" \
  -e CUDA_VISIBLE_DEVICES=0,1,2 \
  -v "${CTRL}:${CTRL}" \
  -v "${DATA}:${DATA}" \
  -w "${REPO}" \
  --shm-size=16g \
  "${IMAGE}" \
  python3 "${CTRL}/ICLR2026/scripts/mae_continued_ssl_psma.py" \
    --fdg-mae-ckpt "${FDG_MAE_CKPT}" \
    --psma-splits-json "${PSMA_SPLITS}" \
    --fdg-splits-json "${FDG_SPLITS}" \
    --psma-cache-dir "${PSMA_CACHE}" \
    --fdg-cache-dir "${FDG_CACHE}" \
    --out-dir "${OUT_SSL}" \
    --epochs "${SSL_EPOCHS}" \
    --batch-size "${SSL_BATCH_SIZE}" \
    --align-weight "${ALIGN_W}" \
    --num-workers "${WORKERS_TRAIN}" \
    --loss-png "${SSL_PNG}" \
    "${FRESH_SSL_FLAG[@]}" \
  >"${SSL_LOG}" 2>&1
SSL_RC=$?
set -e
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
[[ "${SSL_RC}" -eq 0 ]] || { echo "[error] SSL failed rc=${SSL_RC}; see ${SSL_LOG}" >&2; exit "${SSL_RC}"; }
[[ -f "${SSL_LATEST}" ]] || { echo "[error] missing SSL latest: ${SSL_LATEST}" >&2; exit 1; }

# ---------- Stage B: few-shot finetune ----------
echo "[mae-ssl-fs] Stage B few-shot10 finetune …"
FRESH_FT_FLAG=()
[[ "${FRESH_FT}" == "1" ]] && FRESH_FT_FLAG=(--fresh)

FT_LOG="${LOG_DIR}/nohup_mae_psma_fewshot10_${STAMP}.log"
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true

nohup docker run --rm \
  --name "mae_fs10_${STAMP}" \
  --gpus "\"${DOCKER_GPUS}\"" \
  -e CUDA_VISIBLE_DEVICES=0,1,2 \
  -v "${CTRL}:${CTRL}" \
  -v "${DATA}:${DATA}" \
  -w "${REPO}" \
  --shm-size=16g \
  "${IMAGE}" \
  python3 "${CTRL}/ICLR2026/scripts/mae_finetune_fdg_swinbase.py" \
    --cache-dir "${PSMA_CACHE}" \
    --splits-json "${FEWSHOT_SPLITS}" \
    --foundation-ckpt "${SSL_LATEST}" \
    --out-dir "${OUT_FT}" \
    --epochs "${FT_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --sw-batch-size 2 \
    --val-interval 20 \
    --num-workers "${WORKERS_TRAIN}" \
    --cross-val-json "" \
    --psma-val-json "" \
    --late-dual-epochs "${LATE_DUAL_EPOCHS}" \
    --title-tag "MAE PSMA fewshot10 (after SSL)" \
    --ckpt-stem "seg_psma_fewshot10" \
    --cross-val-label "FDG_val_loss" \
    --loss-png "${FT_PNG}" \
    "${FRESH_FT_FLAG[@]}" \
  >"${FT_LOG}" 2>&1 &

echo $! > "${OUT_FT}/nohup.pid"
echo "[mae-ssl-fs] fewshot nohup_pid=$(cat "${OUT_FT}/nohup.pid") container=mae_fs10_${STAMP}"
echo "[mae-ssl-fs] ssl_log=${SSL_LOG}"
echo "[mae-ssl-fs] ft_log=${FT_LOG}"
echo "[mae-ssl-fs] ssl_png=${SSL_PNG}"
echo "[mae-ssl-fs] ft_png=${FT_PNG}"
echo "[mae-ssl-fs] stop_ft: docker stop mae_fs10_${STAMP}"
echo "STAMP=${STAMP}" > "${LOG_DIR}/iclr2026_mae_psma_ssl_fewshot10_${STAMP}.txt"
echo "SSL_LATEST=${SSL_LATEST}" >> "${LOG_DIR}/iclr2026_mae_psma_ssl_fewshot10_${STAMP}.txt"
echo "OUT_FT=${OUT_FT}" >> "${LOG_DIR}/iclr2026_mae_psma_ssl_fewshot10_${STAMP}.txt"

sleep 3
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true
echo "STAMP=${STAMP}"
