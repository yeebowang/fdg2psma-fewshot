#!/usr/bin/env bash
# SegAnyPET full aligned flow · exclusive 3-GPU (0,1,3):
#   FDG: global_bs=6 (=2/GPU DP) · PSMA fewshot bs=2 · 1 fold/GPU · TEST20 1 fold/GPU
set -euo pipefail

CTRL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"
DONE_MARK="${TASK1_SEGANY_DONE_MARK:-${LOG_DIR}/TASK1_SEGANY_PIPELINE_DONE.txt}"

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"

# --- FDG · 3 GPUs · gbs=6 ---
FDG_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_fdg_3gpu_bs6_gpu013"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}"
export TASK1_SEGANY_OFFICIAL_GPUS=0,1,3
export TASK1_INNER_CUDA_VISIBLE_DEVICES=0,1,2
export TASK1_PREFLIGHT_GPUS="0 1 3"
export TASK1_SEGANY_FDG_BATCH_SIZE=6
export TASK1_SEGANY_FDG_NO_OOM_FALLBACK=1
export TASK1_SEGANY_FDG_EPOCHS="${TASK1_SEGANY_FDG_EPOCHS:-100}"
export TASK1_SEGANY_WORKERS_PER_GPU="${TASK1_SEGANY_WORKERS_PER_GPU:-6}"
unset TASK1_SEGANY_WORKERS || true
rm -f "${DATA}/task1_train_workspace/01_train_vis/TASK1_TRAIN_STOP_${FDG_STAMP}.txt" || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"seganypet\":{\"fdg_pretrain\":{\"status\":\"running\",\"stamp\":\"${FDG_STAMP}\",\"bs\":6,\"bs_note\":\"global DP 0,1,3 (=2/GPU)\"}}},\"updated_note\":\"seganypet FDG 3gpu\",\"queue\":[\"seganypet.fdg\",\"seganypet.fs+test\",\"dpdnet\",\"nnunet.psma+test\"]}" || true

echo "[seganypet-full] FDG STAMP=${FDG_STAMP} gpus=0,1,3 gbs=6"
bash "${CTRL}/ICLR2026/run/run_seganypet_fdg_pretrain_bg.sh"

SEG_BEST="${REPO}/runs/${FDG_STAMP}/seganypet_fdg/best.pth"
[[ -f "${SEG_BEST}" ]] || SEG_BEST="${REPO}/runs/${FDG_STAMP}/seganypet_fdg/latest.pth"
[[ -f "${SEG_BEST}" ]] || { echo "[error] missing FDG ckpt" >&2; exit 1; }

export TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}"
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

# --- PSMA fewshot · 1 fold / GPU · bs=2 ---
FS_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_fs50_from_fdg_f258_gpu013"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${FS_STAMP}"
export TASK1_SEGANY_CKPT="${SEG_BEST}"
export TASK1_SEGANY_BATCH_SIZE=2
export TASK1_SEGANY_EPOCHS=100
export TASK1_SEGANY_ACCUM=20
export TASK1_SEGANY_LR_MODE=official
export TASK1_SEGANY_CLICK_MAX=21
export TASK1_SEGANY_GPU_LIST="0 1 3"
export TASK1_PREFLIGHT_GPUS="0 1 3"
export TASK1_SEGANY_WORKERS_PER_GPU="${TASK1_SEGANY_WORKERS_PER_GPU:-6}"
unset TASK1_SEGANY_WORKERS || true
rm -f "${DATA}/task1_train_workspace/01_train_vis/TASK1_TRAIN_STOP_${FS_STAMP}.txt" || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"seganypet\":{\"fdg_pretrain\":{\"status\":\"done\",\"stamp\":\"${FDG_STAMP}\",\"best_ckpt\":\"${SEG_BEST}\"},\"psma_fs50_f258\":{\"status\":\"running\",\"stamp\":\"${FS_STAMP}\",\"bs\":2,\"bs_note\":\"1 fold/GPU\",\"foundation\":\"${SEG_BEST}\"}}},\"updated_note\":\"seganypet fewshot F258\"}" || true

echo "[seganypet-full] fewshot STAMP=${FS_STAMP}"
bash "${CTRL}/ICLR2026/run/run_seganypet_fewshot50_f258_bg.sh"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" --ingest-seganypet-stamp "${FS_STAMP}" || true

# --- TEST20 ---
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
export STAMP="${FS_STAMP}"
export TASK1_FOLD_GPUS=2:0,5:1,8:3
export TASK1_TEST_SKIP_DONE=1
echo "[seganypet-full] TEST20 STAMP=${FS_STAMP}"
bash "${CTRL}/ICLR2026/run/run_eval_seganypet_psma_test20_f258_bg.sh"

{
  echo "done_at=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "fdg_stamp=${FDG_STAMP}"
  echo "fs_stamp=${FS_STAMP}"
  echo "status=ok"
} >"${DONE_MARK}"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"updated_note\":\"SegAnyPET FDG+FS+TEST done → next DpDNet\",\"queue\":[\"dpdnet\",\"nnunet.psma+test\"]}" || true

echo "[seganypet-full] ALL DONE mark=${DONE_MARK}"
