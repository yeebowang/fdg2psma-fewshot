#!/usr/bin/env bash
# After MONAI FDG (1GPU bs=6) finishes on GPU0:
#   1) PSMA fewshot50 F258 · bs=2 · 3 parallel containers all on GPU0
#   2) TEST20 · 1 container/fold on GPU0 · no per-fold shard split
set -euo pipefail

CTRL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"

FDG_STAMP="${TASK1_MONAI_FDG_STAMP:-20260816_214921_iclr2026_monai_fdg_swinvit_1gpu_bs6_tr70_val10_100ep}"
FDG_CNAME="monai_fdg_${FDG_STAMP}"
FDG_BEST="${REPO}/runs/${FDG_STAMP}/best_seg_fdg_monai.pth"
FDG_LATEST="${REPO}/runs/${FDG_STAMP}/latest_seg_fdg_monai.pth"
POLL_SEC="${TASK1_CHAIN_POLL_SEC:-60}"

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"

echo "[chain] wait FDG container=${FDG_CNAME} stamp=${FDG_STAMP}"
while docker ps --format '{{.Names}}' | grep -qx "${FDG_CNAME}"; do
  echo "[chain] FDG still running $(TZ=Asia/Shanghai date +%H:%M:%S)"
  sleep "${POLL_SEC}"
done
echo "[chain] FDG container gone $(TZ=Asia/Shanghai date +%H:%M:%S)"

# Prefer best; fall back to latest if training exited without val-driven best.
FOUNDATION=""
for cand in "${FDG_BEST}" "${FDG_LATEST}"; do
  if [[ -f "${cand}" ]]; then
    FOUNDATION="${cand}"
    break
  fi
done
if [[ -z "${FOUNDATION}" ]]; then
  echo "[chain][error] no FDG ckpt under runs/${FDG_STAMP}" >&2
  exit 1
fi
echo "[chain] foundation=${FOUNDATION}"

# Stage boundary: disarm FDG crash window before fewshot launch.
export TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}"
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

FS_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_monai_psma_fs50_from_fdg_seg_f258_gpu0x3_bs2"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${FS_STAMP}"
export TASK1_MONAI_FDG_SEG_CKPT="${FOUNDATION}"
export TASK1_MAE_FOUNDATION_KIND=seg
export TASK1_MAE_BATCH_SIZE=2
export TASK1_MAE_NUM_EPOCHS="${TASK1_MAE_NUM_EPOCHS:-100}"
export TASK1_MAE_FEWSHOT_FOLDS_CSV=2,5,8
# 3 parallel containers, all pinned to physical GPU 0
export TASK1_MAE_FT_GPU_LIST="0 0 0"
export TASK1_MAE_SEQ_GPUS=0
export TASK1_MAE_F258_FORCE_SEQ=0
export TASK1_PREFLIGHT_GPUS=0
export TASK1_DOCKER_GPUS=0

echo "[chain] start fewshot STAMP=${FS_STAMP} bs=2 gpu_list='0 0 0'"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"monai_swinvit\":{\"psma_fs50_f258\":{\"status\":\"queued\",\"stamp\":\"${FS_STAMP}\",\"bs\":2,\"bs_note\":\"3×parallel on GPU0\",\"total_epochs\":${TASK1_MAE_NUM_EPOCHS}}}},\"updated_note\":\"monai chain: fewshot queued after FDG\"}" || true

bash "${CTRL}/ICLR2026/run/run_monai_psma_fewshot50_f258_from_fdg_seg_bg.sh"
echo "[chain] fewshot DONE ${FS_STAMP}"

# Enter TEST: disarm fewshot arm window first
export TASK1_NNUNET_RESULTS_STAMP_NAME="${FS_STAMP}"
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

echo "[chain] start TEST20 METHOD=monai STAMP=${FS_STAMP} folds→GPU0 (no shard)"
export METHOD=monai
export STAMP="${FS_STAMP}"
export TASK1_FOLD_GPUS=2:0,5:0,8:0
export TASK1_TEST_SKIP_DONE=1
bash "${CTRL}/ICLR2026/run/run_eval_psma_test20_f258_bg.sh"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD_JSON}" --plot-only || true
echo "[chain] ALL DONE FDG=${FDG_STAMP} → FS/TEST=${FS_STAMP}"
