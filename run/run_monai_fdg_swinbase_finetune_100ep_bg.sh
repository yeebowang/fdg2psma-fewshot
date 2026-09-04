#!/usr/bin/env bash
# MONAI SwinViT (Tang SSL) → FDG supervised segmentation (align nnUNet Baseline1 FDG).
# Default: global bs=6 on GPUs 0,1,3. Single-GPU: TASK1_CUDA_VISIBLE_DEVICES=0 TASK1_MAE_BATCH_SIZE=6.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"

TOTAL_EPOCHS="${TASK1_MAE_NUM_EPOCHS:-100}"
BATCH_SIZE="${TASK1_MAE_BATCH_SIZE:-6}"
GPUS="${TASK1_CUDA_VISIBLE_DEVICES:-0,1,3}"
DOCKER_GPUS="${TASK1_DOCKER_GPUS:-device=${GPUS}}"
# Inside container: map N docker GPUs to 0..N-1 (override with TASK1_INNER_CUDA_VISIBLE_DEVICES)
if [[ -n "${TASK1_INNER_CUDA_VISIBLE_DEVICES:-}" ]]; then
  INNER_CVD="${TASK1_INNER_CUDA_VISIBLE_DEVICES}"
else
  _ng="$(awk -F',' '{print NF}' <<<"${GPUS}")"
  INNER_CVD="$(seq -s, 0 $((_ng - 1)))"
fi
if [[ -n "${TASK1_PREFLIGHT_GPUS:-}" ]]; then
  PREFLIGHT_GPUS="${TASK1_PREFLIGHT_GPUS}"
else
  PREFLIGHT_GPUS="${GPUS//,/ }"
fi
WORKERS_TRAIN="${TASK1_MAE_TRAIN_WORKERS:-8}"
VAL_INTERVAL="${TASK1_MAE_VAL_INTERVAL:-20}"
LATE_DUAL_EPOCHS="${TASK1_MAE_LATE_DUAL_EPOCHS:-20}"

SPLITS_JSON="${TASK1_SPLITS_FINAL_JSON:-${CTRL}/ICLR2026/data/splits_baseline1_fdg_nnunet.json}"
PSMA_VAL_JSON="${TASK1_PSMA_VAL_CASES_JSON:-${CTRL}/ICLR2026/data/splits_baseline1_psma_val.json}"
CACHE_DIR="${TASK1_MAE_CACHE_DIR:-${DATA}/task1_train_workspace/mae_cache/fdg_baseline1_70_10}"
PSMA_CACHE_DIR="${TASK1_MAE_PSMA_CACHE_DIR:-${DATA}/task1_train_workspace/mae_cache/psma_baseline2_70_10}"
FOUNDATION="${TASK1_MAE_GENERIC_CKPT:-${REPO}/weights/generic/model_swinvit.pt}"
FOUNDATION_KIND="${TASK1_MAE_FOUNDATION_KIND:-monai_swinvit}"
BOARD_METHOD="${TASK1_BOARD_METHOD:-monai_swinvit}"
if [[ "${FOUNDATION_KIND}" == "none" ]]; then
  BOARD_METHOD="${TASK1_BOARD_METHOD:-monai_scratch}"
fi
DEPTHS="${TASK1_MAE_DEPTHS:-2,2,2,2}"
USE_V2="${TASK1_MAE_USE_V2:-0}"
BB_LR_MULT="${TASK1_MAE_BACKBONE_LR_MULT:-0.1}"
FREEZE_ENC_EP="${TASK1_MAE_FREEZE_ENCODER_EPOCHS:-20}"
if [[ "${FOUNDATION_KIND}" == "none" ]]; then
  BB_LR_MULT="${TASK1_MAE_BACKBONE_LR_MULT:-1.0}"
  FREEZE_ENC_EP="${TASK1_MAE_FREEZE_ENCODER_EPOCHS:-0}"
fi

LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"

STAMP_TZ="${TASK1_STAMP_TZ:-Asia/Shanghai}"
if [[ -n "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME}"
elif [[ "${FOUNDATION_KIND}" == "none" || "${BOARD_METHOD}" == "monai_scratch" ]]; then
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_monai_scratch_fdg_swinvit_gpu013_bs${BATCH_SIZE}_tr70_val10_${TOTAL_EPOCHS}ep"
else
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_monai_fdg_swinvit_gpu013_bs${BATCH_SIZE}_tr70_val10_${TOTAL_EPOCHS}ep"
fi
OUT_DIR="${TASK1_MAE_OUT_DIR:-${REPO}/runs/${STAMP}}"
LOSS_PNG="${TASK1_LOSS_OUT_NAME:-${LOG_DIR}/loss_curve_iclr2026_monai_fdg_${STAMP}.png}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${CACHE_DIR}" "${PSMA_CACHE_DIR}"

TITLE_TAG="MONAI SwinViT FDG supervised"
if [[ "${FOUNDATION_KIND}" == "none" ]]; then
  TITLE_TAG="MONAI SwinViT scratch FDG"
fi
echo "[monai-fdg] STAMP=${STAMP}"
echo "[monai-fdg] foundation=${FOUNDATION} kind=${FOUNDATION_KIND} depths=${DEPTHS} use_v2=${USE_V2}"
echo "[monai-fdg] out=${OUT_DIR} ep=${TOTAL_EPOCHS} bs=${BATCH_SIZE} gpus=${GPUS} inner_cvd=${INNER_CVD}"

if [[ "${FOUNDATION_KIND}" != "none" ]]; then
  [[ -f "${FOUNDATION}" ]] || { echo "[error] missing foundation: ${FOUNDATION}" >&2; exit 1; }
fi
[[ -f "${SPLITS_JSON}" ]] || { echo "[error] missing splits: ${SPLITS_JSON}" >&2; exit 1; }

export TASK1_PREFLIGHT_GPUS="${PREFLIGHT_GPUS}"
export TASK1_PREFLIGHT_LABEL="iclr2026-monai-fdg"
bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true

export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_BASE="${DATA}"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"fdg_pretrain\":{\"status\":\"running\",\"stamp\":\"${STAMP}\",\"foundation\":\"${FOUNDATION}\",\"bs\":${BATCH_SIZE},\"bs_note\":\"global 2×3GPU\",\"total_epochs\":${TOTAL_EPOCHS}}}},\"updated_note\":\"${BOARD_METHOD} fdg start\"}" || true

bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

NOHUP_LOG="${LOG_DIR}/nohup_monai_fdg_${STAMP}.log"
echo "[monai-fdg] launching -> ${NOHUP_LOG}"

FRESH_FLAG=()
if [[ "${TASK1_MAE_FRESH:-1}" == "1" ]]; then
  FRESH_FLAG=(--fresh)
else
  echo "[monai-fdg] resume mode (TASK1_MAE_FRESH=0) — keep latest under ${OUT_DIR}"
fi

# Wait for container (foreground via nohup+wait in caller preferred; here block with wait)
docker rm -f "monai_fdg_${STAMP}" >/dev/null 2>&1 || true
nohup docker run --rm \
  --name "monai_fdg_${STAMP}" \
  --gpus "\"${DOCKER_GPUS}\"" \
  -e CUDA_VISIBLE_DEVICES="${INNER_CVD}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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
    --depths "${DEPTHS}" \
    --use-v2 "${USE_V2}" \
    --out-dir "${OUT_DIR}" \
    --epochs "${TOTAL_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --sw-batch-size 2 \
    --val-interval "${VAL_INTERVAL}" \
    --num-workers "${WORKERS_TRAIN}" \
    --backbone-lr-mult "${BB_LR_MULT}" \
    --freeze-encoder-epochs "${FREEZE_ENC_EP}" \
    --psma-val-json "${PSMA_VAL_JSON}" \
    --psma-cache-dir "${PSMA_CACHE_DIR}" \
    --late-dual-epochs "${LATE_DUAL_EPOCHS}" \
    --title-tag "${TITLE_TAG}" \
    --ckpt-stem "seg_fdg_monai" \
    --loss-png "${LOSS_PNG}" \
    "${FRESH_FLAG[@]}" \
  >"${NOHUP_LOG}" 2>&1 &
echo $! > "${OUT_DIR}/nohup.pid"
sleep 3
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true

set +e
wait "$(cat "${OUT_DIR}/nohup.pid")"
RC=$?
set -e
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

BEST="${OUT_DIR}/best_seg_fdg_monai.pth"
[[ -f "${BEST}" ]] || BEST="${OUT_DIR}/latest_seg_fdg_monai.pth"
# Treat usable ckpt as success even if process OOM'd later (continue/pipeline can proceed)
if [[ -f "${BEST}" ]]; then
  if [[ "${RC}" -eq 0 ]]; then
    ST="done"
  else
    ST="done"
    NOTE="exited rc=${RC} but ckpt present"
  fi
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" \
    --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"fdg_pretrain\":{\"status\":\"${ST}\",\"stamp\":\"${STAMP}\",\"best_ckpt\":\"${BEST}\",\"note\":\"${NOTE:-}\"}}},\"updated_note\":\"${BOARD_METHOD} fdg ${ST}\"}" || true
else
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" \
    --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"fdg_pretrain\":{\"status\":\"failed\",\"stamp\":\"${STAMP}\"}}},\"updated_note\":\"${BOARD_METHOD} fdg failed\"}" || true
fi

echo "STAMP=${STAMP}" > "${LOG_DIR}/iclr2026_monai_fdg_${STAMP}.txt"
echo "BEST=${BEST}" >> "${LOG_DIR}/iclr2026_monai_fdg_${STAMP}.txt"
echo "[monai-fdg] done rc=${RC} best=${BEST}"
exit "${RC}"
