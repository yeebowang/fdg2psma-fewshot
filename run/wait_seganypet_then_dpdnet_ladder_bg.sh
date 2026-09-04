#!/usr/bin/env bash
# After SegAnyPET FDG+fewshot+TEST done, wait until ≥2 of GPUs {0,1,3} are free,
# then launch DpDNet FDG OOM ladder (1×bs6 → 2×≈3 → 3×≈2).
set -euo pipefail

CTRL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"
DONE_MARK="${TASK1_SEGANY_DONE_MARK:-${LOG_DIR}/TASK1_SEGANY_PIPELINE_DONE.txt}"
POLL_SEC="${TASK1_CHAIN_POLL_SEC:-60}"
POOL=(0 1 3)
NEED_FREE="${TASK1_DPDNET_MIN_FREE_GPUS:-2}"

# Keep old prep-only STOP so accidental retry of old stamp does not resume.
OLD_DPD_STAMP="${TASK1_DPDNET_BLOCK_STAMP:-20260816_220615_iclr2026_dpdnet_fdg_1gpu_bs6_100ep_gpu1}"

_gpu_busy() {
  local g="$1"
  docker ps -q | while read -r id; do
    ids="$(docker inspect -f '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{.}} {{end}}{{end}}' "$id" 2>/dev/null || true)"
    for x in $ids; do
      if [[ "$x" == "$g" ]]; then
        echo 1
        return 0
      fi
    done
  done
  return 1
}

_count_free() {
  local n=0 g
  for g in "${POOL[@]}"; do
    if ! _gpu_busy "$g" >/dev/null; then
      n=$((n + 1))
    fi
  done
  echo "${n}"
}

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"

echo "[wait-dpdnet] need SegAnyPET DONE mark: ${DONE_MARK}"
while [[ ! -f "${DONE_MARK}" ]] || ! grep -q 'status=ok' "${DONE_MARK}"; do
  echo "[wait-dpdnet] waiting SegAnyPET pipeline… $(TZ=Asia/Shanghai date +%H:%M:%S)"
  sleep "${POLL_SEC}"
done
echo "[wait-dpdnet] SegAnyPET done: $(cat "${DONE_MARK}")"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"dpdnet\":{\"fdg_pretrain\":{\"status\":\"queued\",\"note\":\"waiting ≥${NEED_FREE} free among GPU0/1/3 after SegAnyPET\"}}},\"updated_note\":\"DpDNet queued after SegAnyPET\"}" || true

echo "[wait-dpdnet] need ≥${NEED_FREE} free among ${POOL[*]}"
while true; do
  n="$(_count_free)"
  if [[ "${n}" -ge "${NEED_FREE}" ]]; then
    echo "[wait-dpdnet] free_count=${n} → launch ladder $(TZ=Asia/Shanghai date +%H:%M:%S)"
    break
  fi
  echo "[wait-dpdnet] free_count=${n} < ${NEED_FREE} $(TZ=Asia/Shanghai date +%H:%M:%S)"
  sleep "${POLL_SEC}"
done

# ensure old stamp stays stopped
mkdir -p "${DATA}/task1_train_workspace/01_train_vis"
if [[ ! -f "${DATA}/task1_train_workspace/01_train_vis/TASK1_TRAIN_STOP_${OLD_DPD_STAMP}.txt" ]]; then
  echo "reason=block_old_stamp_until_ladder" \
    >"${DATA}/task1_train_workspace/01_train_vis/TASK1_TRAIN_STOP_${OLD_DPD_STAMP}.txt"
fi

export TASK1_DPDNET_GPU_POOL=0,1,3
export TASK1_DPDNET_BATCH_SIZE=6
export TASK1_DPDNET_NUM_EPOCHS="${TASK1_DPDNET_NUM_EPOCHS:-100}"
export TASK1_DPDNET_SKIP_PREPARE=1

bash "${CTRL}/ICLR2026/run/run_dpdnet_fdg_bs_ladder_bg.sh"
echo "[wait-dpdnet] ladder finished"
