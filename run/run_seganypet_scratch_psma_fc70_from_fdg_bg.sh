#!/usr/bin/env bash
# SegAnyPET scratch · PSMA fc70% (1 GPU) · FDG-scratch click init → TEST20
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
VIS="${CTRL}/ICLR2026/vis"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
GPU="${TASK1_PSMA_FC70_GPU:-${TASK1_CUDA_VISIBLE_DEVICES:-0}}"
GPU="${GPU%%,*}"

export TASK1_BOARD_METHOD="${TASK1_BOARD_METHOD:-seganypet_scratch}"
export TASK1_PSMA_FC70_GPU="${GPU}"
export TASK1_CUDA_VISIBLE_DEVICES="${GPU}"

if [[ -z "${TASK1_SEGANY_CKPT:-}" ]]; then
  STAMP=""
  if [[ -f "${VIS}/seganypet_scratch_fdg_LAST_STAMP.txt" ]]; then
    STAMP="$(head -n1 "${VIS}/seganypet_scratch_fdg_LAST_STAMP.txt" | tr -d '[:space:]')"
  fi
  for cand in \
    "${REPO}/runs/${STAMP}/seganypet_fdg/best.pth" \
    "${REPO}/runs/${STAMP}/seganypet_fdg/latest.pth"
  do
    if [[ -n "${STAMP}" && -f "${cand}" ]]; then
      export TASK1_SEGANY_CKPT="${cand}"
      break
    fi
  done
fi
[[ -n "${TASK1_SEGANY_CKPT:-}" && -f "${TASK1_SEGANY_CKPT}" ]] || {
  echo "[error] seganypet-scratch fc70: missing FDG scratch ckpt" >&2
  exit 1
}

if [[ -z "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  export TASK1_NNUNET_RESULTS_STAMP_NAME="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_scratch_psma_fc70_gpu${GPU}"
fi

echo "[seganypet-scratch-fc70] gpu=${GPU} ckpt=${TASK1_SEGANY_CKPT} stamp=${TASK1_NNUNET_RESULTS_STAMP_NAME}"

bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
trap 'bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true' EXIT

bash "${CTRL}/ICLR2026/run/run_seganypet_psma_fc70_from_fdg_bg.sh"
