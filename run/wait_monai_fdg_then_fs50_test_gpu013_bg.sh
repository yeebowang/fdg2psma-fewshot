#!/usr/bin/env bash
# Wait MONAI FDG (resume stamp) → PSMA fewshot F258 · 1GPU/fold on 0,1,3 · then TEST20 no-shard
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
# First wait until FDG is actually running (avoid race if chain starts before docker)
for _i in $(seq 1 120); do
  if docker ps --format '{{.Names}}' | grep -qx "${FDG_CNAME}"; then
    echo "[chain] FDG container seen $(TZ=Asia/Shanghai date +%H:%M:%S)"
    break
  fi
  echo "[chain] waiting FDG to appear… $(TZ=Asia/Shanghai date +%H:%M:%S)"
  sleep 5
done
if ! docker ps --format '{{.Names}}' | grep -qx "${FDG_CNAME}"; then
  # If already finished with ckpt, skip wait-loop and proceed
  if [[ -f "${FDG_BEST}" || -f "${FDG_LATEST}" ]]; then
    echo "[chain] FDG container not running but ckpt exists → continue to fewshot"
  else
    echo "[chain][error] FDG never started and no ckpt" >&2
    exit 1
  fi
fi
while docker ps --format '{{.Names}}' | grep -qx "${FDG_CNAME}"; do
  echo "[chain] FDG still running $(TZ=Asia/Shanghai date +%H:%M:%S)"
  sleep "${POLL_SEC}"
done
echo "[chain] FDG container gone $(TZ=Asia/Shanghai date +%H:%M:%S)"

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

export TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}"
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

FS_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_monai_psma_fs50_from_fdg_seg_f258_gpu013_bs2"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${FS_STAMP}"
export TASK1_MONAI_FDG_SEG_CKPT="${FOUNDATION}"
export TASK1_MAE_FOUNDATION_KIND=seg
export TASK1_MAE_BATCH_SIZE=2
export TASK1_MAE_NUM_EPOCHS="${TASK1_MAE_NUM_EPOCHS:-100}"
export TASK1_MAE_FEWSHOT_FOLDS_CSV=2,5,8
# F2→0 F5→1 F8→3
export TASK1_MAE_FT_GPU_LIST="0 1 3"
export TASK1_MAE_SEQ_GPUS=0,1,3
export TASK1_MAE_F258_FORCE_SEQ=0
export TASK1_PREFLIGHT_GPUS="0 1 3"
# clear fewshot STOP if present
rm -f "${DATA}/task1_train_workspace/01_train_vis/TASK1_TRAIN_STOP_${FS_STAMP}.txt" || true

echo "[chain] start fewshot STAMP=${FS_STAMP} bs=2 gpus=0,1,3"
bash "${CTRL}/ICLR2026/run/run_monai_psma_fewshot50_f258_from_fdg_seg_bg.sh"
echo "[chain] fewshot DONE ${FS_STAMP}"

bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

echo "[chain] start TEST20 METHOD=monai STAMP=${FS_STAMP} folds→0/1/3 (no shard)"
export METHOD=monai
export STAMP="${FS_STAMP}"
export TASK1_FOLD_GPUS=2:0,5:1,8:3
export TASK1_TEST_SKIP_DONE=1
bash "${CTRL}/ICLR2026/run/run_eval_psma_test20_f258_bg.sh"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD_JSON}" --plot-only || true

DONE_MARK="${TASK1_MONAI_DONE_MARK:-${LOG_DIR}/TASK1_MONAI_PIPELINE_DONE.txt}"
{
  echo "done_at=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "fdg_stamp=${FDG_STAMP}"
  echo "fs_stamp=${FS_STAMP}"
  echo "status=ok"
} >"${DONE_MARK}"
echo "[chain] ALL DONE FDG=${FDG_STAMP} → FS/TEST=${FS_STAMP} mark=${DONE_MARK}"
