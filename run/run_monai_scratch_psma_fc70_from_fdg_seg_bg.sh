#!/usr/bin/env bash
# MONAI SwinViT scratch · PSMA fc70% (1 GPU) · FDG-scratch seg init → TEST20
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
VIS="${CTRL}/ICLR2026/vis"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
GPU="${TASK1_PSMA_FC70_GPU:-${TASK1_CUDA_VISIBLE_DEVICES:-0}}"
GPU="${GPU%%,*}"

export TASK1_BOARD_METHOD="${TASK1_BOARD_METHOD:-monai_scratch}"
export TASK1_FC70_EVAL_METHOD="${TASK1_FC70_EVAL_METHOD:-monai_scratch}"
export TASK1_PSMA_FC70_GPU="${GPU}"
export TASK1_CUDA_VISIBLE_DEVICES="${GPU}"

if [[ -z "${TASK1_MONAI_FDG_SEG_CKPT:-}" ]]; then
  STAMP=""
  if [[ -f "${VIS}/monai_scratch_fdg_LAST_STAMP.txt" ]]; then
    STAMP="$(head -n1 "${VIS}/monai_scratch_fdg_LAST_STAMP.txt" | tr -d '[:space:]')"
  fi
  for cand in \
    "${REPO}/runs/${STAMP}/best_seg_fdg_monai.pth" \
    "${REPO}/runs/${STAMP}/latest_seg_fdg_monai.pth"
  do
    if [[ -n "${STAMP}" && -f "${cand}" ]]; then
      export TASK1_MONAI_FDG_SEG_CKPT="${cand}"
      break
    fi
  done
fi
[[ -n "${TASK1_MONAI_FDG_SEG_CKPT:-}" && -f "${TASK1_MONAI_FDG_SEG_CKPT}" ]] || {
  echo "[error] monai-scratch fc70: missing FDG scratch ckpt" >&2
  exit 1
}

if [[ -z "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  export TASK1_NNUNET_RESULTS_STAMP_NAME="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_monai_scratch_psma_fc70_from_fdg_seg_gpu${GPU}"
fi

echo "[monai-scratch-fc70] gpu=${GPU} ckpt=${TASK1_MONAI_FDG_SEG_CKPT} stamp=${TASK1_NNUNET_RESULTS_STAMP_NAME}"

bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
trap 'bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true' EXIT

bash "${CTRL}/ICLR2026/run/run_monai_psma_fc70_from_fdg_seg_bg.sh"
