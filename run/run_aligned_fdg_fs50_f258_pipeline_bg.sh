#!/usr/bin/env bash
# Master chain: align MAE / MONAI / SegAnyPET to nnUNet protocol
#   FDG supervised segmentation → PSMA fewshot50 f2/5/8
#
# Queue (GPUs 0,1,3 exclusive per stage):
#   1) MAE fewshot from existing FDG seg (FDG already done)
#   2) MONAI FDG supervised → MONAI fewshot
#   3) SegAnyPET FDG click pretrain → SegAnyPET fewshot (from FDG best)
#
# Progress board: ICLR2026/vis/iclr2026_aligned_fdg_fs50_f258_board.json + .png
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"
BOARD_PNG="${TASK1_ALIGN_BOARD_PNG:-${LOG_DIR}/progress_iclr2026_aligned_fdg_fs50_f258_board.png}"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"
export TASK1_BASE="${DATA}"

PIPE_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_aligned_fdg_fs50_f258_pipeline"
PIPE_LOG="${LOG_DIR}/nohup_aligned_fdg_fs50_pipeline_${PIPE_STAMP}.log"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${PIPE_LOG}") 2>&1
echo "[aligned-pipe] PIPE_STAMP=${PIPE_STAMP}"
echo "[aligned-pipe] board=${BOARD_JSON}"

# init board + start watcher
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" --png "${BOARD_PNG}" --init

nohup python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" --png "${BOARD_PNG}" --watch 60 \
  >"${LOG_DIR}/nohup_aligned_board_watch_${PIPE_STAMP}.log" 2>&1 &
echo $! > "${LOG_DIR}/aligned_board_watch_${PIPE_STAMP}.pid"
echo "[aligned-pipe] board watcher pid=$(cat "${LOG_DIR}/aligned_board_watch_${PIPE_STAMP}.pid")"

# crash guard (require_arm)
bash "${CTRL}/run_task1_train_auto_resume_guard_bg.sh" || true

# ---------- 1) MAE fewshot from FDG seg ----------
MAE_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_mae_psma_fs50_from_fdg_seg_f258_gpu013"
echo "[aligned-pipe] === STAGE MAE fewshot ${MAE_STAMP} ==="
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"updated_note\":\"queue: MAE fewshot\",\"queue\":[\"mae_swinunetr.psma_fs50_f258 (running)\",\"monai_swinvit.fdg_pretrain\",\"monai_swinvit.psma_fs50_f258\",\"seganypet.fdg_pretrain\",\"seganypet.psma_fs50_f258\"]}" || true

TASK1_NNUNET_RESULTS_STAMP_NAME="${MAE_STAMP}" \
  bash "${CTRL}/ICLR2026/run/run_mae_psma_fewshot50_f258_from_fdg_seg_bg.sh"

# ---------- 2) MONAI FDG → fewshot ----------
MONAI_FDG_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_monai_fdg_swinvit_gpu013_bs6_tr70_val10_100ep"
echo "[aligned-pipe] === STAGE MONAI FDG ${MONAI_FDG_STAMP} ==="
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"updated_note\":\"queue: MONAI FDG\",\"queue\":[\"mae_swinunetr.psma_fs50_f258 (done)\",\"monai_swinvit.fdg_pretrain (running)\",\"monai_swinvit.psma_fs50_f258\",\"seganypet.fdg_pretrain\",\"seganypet.psma_fs50_f258\"]}" || true

TASK1_NNUNET_RESULTS_STAMP_NAME="${MONAI_FDG_STAMP}" \
  bash "${CTRL}/ICLR2026/run/run_monai_fdg_swinbase_finetune_100ep_bg.sh"

MONAI_BEST="${REPO}/runs/${MONAI_FDG_STAMP}/best_seg_fdg_monai.pth"
[[ -f "${MONAI_BEST}" ]] || MONAI_BEST="${REPO}/runs/${MONAI_FDG_STAMP}/latest_seg_fdg_monai.pth"
[[ -f "${MONAI_BEST}" ]] || { echo "[error] missing MONAI FDG best" >&2; exit 1; }

MONAI_FS_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_monai_psma_fs50_from_fdg_seg_f258_gpu013"
echo "[aligned-pipe] === STAGE MONAI fewshot ${MONAI_FS_STAMP} ==="
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"updated_note\":\"queue: MONAI fewshot\",\"queue\":[\"mae done\",\"monai fdg done\",\"monai_swinvit.psma_fs50_f258 (running)\",\"seganypet.fdg_pretrain\",\"seganypet.psma_fs50_f258\"]}" || true

TASK1_MONAI_FDG_SEG_CKPT="${MONAI_BEST}" \
TASK1_NNUNET_RESULTS_STAMP_NAME="${MONAI_FS_STAMP}" \
  bash "${CTRL}/ICLR2026/run/run_monai_psma_fewshot50_f258_from_fdg_seg_bg.sh"

# ---------- 3) SegAnyPET FDG → fewshot ----------
SEG_FDG_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_fdg_pretrain_gpu013"
echo "[aligned-pipe] === STAGE SegAnyPET FDG ${SEG_FDG_STAMP} ==="
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"updated_note\":\"queue: SegAnyPET FDG\",\"queue\":[\"mae/monai done\",\"seganypet.fdg_pretrain (running)\",\"seganypet.psma_fs50_f258\"]}" || true

TASK1_SEGANY_FDG_BATCH_SIZE="${TASK1_SEGANY_FDG_BATCH_SIZE:-6}" \
TASK1_NNUNET_RESULTS_STAMP_NAME="${SEG_FDG_STAMP}" \
  bash "${CTRL}/ICLR2026/run/run_seganypet_fdg_pretrain_bg.sh"

SEG_BEST="${REPO}/runs/${SEG_FDG_STAMP}/seganypet_fdg/best.pth"
[[ -f "${SEG_BEST}" ]] || SEG_BEST="${REPO}/runs/${SEG_FDG_STAMP}/seganypet_fdg/latest.pth"
[[ -f "${SEG_BEST}" ]] || { echo "[error] missing SegAnyPET FDG best" >&2; exit 1; }

SEG_FS_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_fs50_from_fdg_f258_gpu013"
echo "[aligned-pipe] === STAGE SegAnyPET fewshot ${SEG_FS_STAMP} ==="
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"seganypet\":{\"psma_fs50_f258\":{\"status\":\"running\",\"stamp\":\"${SEG_FS_STAMP}\",\"foundation\":\"${SEG_BEST}\",\"bs\":2,\"bs_note\":\"per-GPU\"}}},\"updated_note\":\"queue: SegAnyPET fewshot bs=2\"}" || true

# PSMA fewshot: bs=2 · 1GPU/fold parallel (align MAE/MONAI); official lr schedule
if [[ -f "${CTRL}/ICLR2026/run/run_seganypet_fewshot50_f258_bg.sh" ]]; then
  TASK1_SEGANY_CKPT="${SEG_BEST}" \
  TASK1_SEGANY_BATCH_SIZE="${TASK1_SEGANY_BATCH_SIZE:-2}" \
  TASK1_SEGANY_EPOCHS="${TASK1_SEGANY_EPOCHS:-100}" \
  TASK1_SEGANY_ACCUM="${TASK1_SEGANY_ACCUM:-20}" \
  TASK1_SEGANY_LR_MODE=official \
  TASK1_SEGANY_LR="${TASK1_SEGANY_LR:-8e-4}" \
  TASK1_SEGANY_MILESTONES="${TASK1_SEGANY_MILESTONES:-60,85}" \
  TASK1_SEGANY_CLICK_MAX="${TASK1_SEGANY_CLICK_MAX:-21}" \
  TASK1_NNUNET_RESULTS_STAMP_NAME="${SEG_FS_STAMP}" \
    bash "${CTRL}/ICLR2026/run/run_seganypet_fewshot50_f258_bg.sh"
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" --ingest-seganypet-stamp "${SEG_FS_STAMP}" || true
fi

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"updated_note\":\"ALL STAGES DONE\",\"queue\":[]}" || true

echo "[aligned-pipe] ALL DONE pipe=${PIPE_STAMP}"
echo "board=${BOARD_JSON}"
echo "png=${BOARD_PNG}"
echo "log=${PIPE_LOG}"
