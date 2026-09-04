#!/usr/bin/env bash
# SegAnyPET full aligned stage on GPU1+3 (FDG), then fewshot F258, then TEST20,
# then write DONE marker for DpDNet ladder.
#
# FDG: global_bs=6 on GPUs 1,3 (≈3/GPU)
# PSMA fewshot: bs=2 · parallel f2/f5/f8 on free GPUs among 0,1,3 (default 0 1 3)
# TEST20: 1 container/fold · no shard
set -euo pipefail

CTRL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"
DONE_MARK="${LOG_DIR}/TASK1_SEGANY_PIPELINE_DONE.txt"
POLL_SEC="${TASK1_CHAIN_POLL_SEC:-45}"
TARGET_GPUS=(1 3)

_gpu_busy() {
  local g="$1"
  docker ps -q | while read -r id; do
    ids="$(docker inspect -f '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{.}} {{end}}{{end}}' "$id" 2>/dev/null || true)"
    for x in $ids; do
      if [[ "$x" == "$g" ]]; then
        docker inspect -f '{{.Name}}' "$id" | sed 's#^/##'
        return 0
      fi
    done
  done
  return 1
}

echo "[seganypet-chain] wait GPU ${TARGET_GPUS[*]} free for FDG"
while true; do
  busy=()
  for g in "${TARGET_GPUS[@]}"; do
    names="$(_gpu_busy "$g" || true)"
    if [[ -n "${names}" ]]; then
      busy+=("GPU${g}:{${names//$'\n'/,}}")
    fi
  done
  if [[ "${#busy[@]}" -eq 0 ]]; then
    echo "[seganypet-chain] GPU1+3 free $(TZ=Asia/Shanghai date +%H:%M:%S)"
    break
  fi
  echo "[seganypet-chain] busy ${busy[*]} $(TZ=Asia/Shanghai date +%H:%M:%S)"
  sleep "${POLL_SEC}"
done

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"

# --- FDG ---
FDG_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_fdg_2gpu_bs6_gpu13"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}"
export TASK1_SEGANY_OFFICIAL_GPUS=1,3
export TASK1_INNER_CUDA_VISIBLE_DEVICES=0,1
export TASK1_PREFLIGHT_GPUS="1 3"
export TASK1_SEGANY_FDG_BATCH_SIZE=6
export TASK1_SEGANY_FDG_NO_OOM_FALLBACK=0
export TASK1_SEGANY_FDG_EPOCHS="${TASK1_SEGANY_FDG_EPOCHS:-100}"
export TASK1_SEGANY_WORKERS_PER_GPU="${TASK1_SEGANY_WORKERS_PER_GPU:-6}"
# 2 GPUs → total DataLoader workers = 6*2 = 12 (unless TASK1_SEGANY_WORKERS set)
unset TASK1_SEGANY_WORKERS || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"seganypet\":{\"fdg_pretrain\":{\"status\":\"running\",\"stamp\":\"${FDG_STAMP}\",\"bs\":6,\"bs_note\":\"global DP 1,3 (=3/GPU)\"}}},\"updated_note\":\"seganypet FDG GPU1+3\"}" || true

echo "[seganypet-chain] FDG STAMP=${FDG_STAMP}"
bash "${CTRL}/ICLR2026/run/run_seganypet_fdg_pretrain_bg.sh"

SEG_BEST="${REPO}/runs/${FDG_STAMP}/seganypet_fdg/best.pth"
[[ -f "${SEG_BEST}" ]] || SEG_BEST="${REPO}/runs/${FDG_STAMP}/seganypet_fdg/latest.pth"
[[ -f "${SEG_BEST}" ]] || { echo "[error] missing FDG ckpt" >&2; exit 1; }

# --- PSMA fewshot ---
export TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}"
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

FS_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_fs50_from_fdg_f258_gpu013"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${FS_STAMP}"
export TASK1_SEGANY_CKPT="${SEG_BEST}"
export TASK1_SEGANY_BATCH_SIZE=2
export TASK1_SEGANY_EPOCHS=100
export TASK1_SEGANY_ACCUM=20
export TASK1_SEGANY_LR_MODE=official
export TASK1_SEGANY_CLICK_MAX=21
export TASK1_SEGANY_GPU_LIST="${TASK1_SEGANY_GPU_LIST:-0 1 3}"
export TASK1_PREFLIGHT_GPUS="${TASK1_SEGANY_GPU_LIST}"
export TASK1_SEGANY_WORKERS_PER_GPU="${TASK1_SEGANY_WORKERS_PER_GPU:-6}"
unset TASK1_SEGANY_WORKERS || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"seganypet\":{\"psma_fs50_f258\":{\"status\":\"running\",\"stamp\":\"${FS_STAMP}\",\"bs\":2,\"bs_note\":\"per-GPU parallel\",\"foundation\":\"${SEG_BEST}\"}}},\"updated_note\":\"seganypet fewshot running\"}" || true

echo "[seganypet-chain] fewshot STAMP=${FS_STAMP}"
bash "${CTRL}/ICLR2026/run/run_seganypet_fewshot50_f258_bg.sh"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" --ingest-seganypet-stamp "${FS_STAMP}" || true

# --- TEST20 ---
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
export STAMP="${FS_STAMP}"
export TASK1_FOLD_GPUS="${TASK1_FOLD_GPUS:-2:0,5:1,8:3}"
export TASK1_TEST_SKIP_DONE=1
echo "[seganypet-chain] TEST20 STAMP=${FS_STAMP}"
bash "${CTRL}/ICLR2026/run/run_eval_seganypet_psma_test20_f258_bg.sh"

{
  echo "done_at=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "fdg_stamp=${FDG_STAMP}"
  echo "fs_stamp=${FS_STAMP}"
  echo "status=ok"
} >"${DONE_MARK}"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"updated_note\":\"SegAnyPET FDG+FS+TEST done → DpDNet ladder may start\"}" || true

echo "[seganypet-chain] ALL DONE marker=${DONE_MARK}"
