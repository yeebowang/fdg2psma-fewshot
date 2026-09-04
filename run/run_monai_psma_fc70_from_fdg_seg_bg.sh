#!/usr/bin/env bash
# MONAI PSMA fc70% · single GPU · tr25 val25 · 100ep → TEST20
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/ICLR2026/run/_psma_fc70_env.sh"
CTRL="${ROOT}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"

STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME:-}"
BOARD_METHOD="${TASK1_BOARD_METHOD:-monai_swinvit}"
[[ -n "${STAMP}" ]] || {
  if [[ "${BOARD_METHOD}" == "monai_scratch" ]]; then
    STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_monai_scratch_psma_fc70_from_fdg_seg_gpu${PSMA_FC70_GPU}"
  else
    STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_monai_psma_fc70_from_fdg_seg_gpu${PSMA_FC70_GPU}"
  fi
}
FOUNDATION="${TASK1_MONAI_FDG_SEG_CKPT:-${REPO}/runs/20260816_214921_iclr2026_monai_fdg_swinvit_1gpu_bs6_tr70_val10_100ep/best_seg_fdg_monai.pth}"
OUT="${REPO}/runs/${STAMP}/monai/fold${PSMA_FC70_FOLD}"
mkdir -p "${OUT}" "${VIS}"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"${PSMA_FC70_STAGE}\":{\"status\":\"running\",\"stamp\":\"${STAMP}\",\"note\":\"fc70% · single run\"}}},\"updated_note\":\"${BOARD_METHOD} fc70 training\"}" || true

bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true
docker run --rm --gpus "device=${PSMA_FC70_GPU}" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
  -w "${REPO}" --shm-size=8g "${IMAGE}" \
  python3 "${CTRL}/ICLR2026/scripts/mae_finetune_fdg_swinbase.py" \
    --cache-dir "${DATA}/task1_train_workspace/mae_cache/psma_baseline2_70_10" \
    --splits-json "${PSMA_FC70_SPLITS}" \
    --foundation-ckpt "${FOUNDATION}" --foundation-kind seg \
    --depths 2,2,2,2 --use-v2 0 \
    --out-dir "${OUT}" --epochs "${PSMA_FC70_EP}" --batch-size "${PSMA_FC70_BS}" \
    --sw-batch-size "${PSMA_FC70_BS}" --val-interval "${PSMA_FC70_VAL_EVERY}" \
    --num-workers 4 --ckpt-stem "seg_psma_fc70_monai_fdgseg_f${PSMA_FC70_FOLD}" --fresh

METHOD="${TASK1_FC70_EVAL_METHOD:-${BOARD_METHOD}}"
if [[ "${METHOD}" == "monai_scratch" ]]; then
  :
elif [[ "${METHOD}" == "monai_swinvit" ]]; then
  METHOD=monai
fi
METHOD="${METHOD}" STAMP="${STAMP}" TASK1_PSMA_BOARD_STAGE="${PSMA_FC70_STAGE}" \
  TASK1_MAE_FEWSHOT_FOLDS_CSV="${PSMA_FC70_FOLD}" TASK1_FOLD_GPUS="${PSMA_FC70_FOLD}:${PSMA_FC70_GPU}" \
  bash "${CTRL}/ICLR2026/run/run_eval_psma_test20_fc70_bg.sh"

echo "[monai-fc70] ALL DONE ${STAMP}"
