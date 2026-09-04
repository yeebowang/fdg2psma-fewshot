#!/usr/bin/env bash
# After MONAI scratch + SegAnyPET scratch 9fold → nnUNet MIM, then DpDNet dual-enc.
#
#   bash ICLR2026/run/run_nnunet_mim_dpdnet_dualenc_after_scratch_queue_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
POLL_SEC="${TASK1_CHAIN_POLL_SEC:-60}"
DONE_MONAI="${VIS}/TASK1_MONAI_SCRATCH_9FOLD_DONE.txt"
DONE_SEG="${VIS}/TASK1_SEGANY_SCRATCH_9FOLD_DONE.txt"
DONE_SELF="${VIS}/TASK1_NNUNET_MIM_DPDNET_DUALENC_DONE.txt"
PID_FILE="${VIS}/nnunet_mim_dpdnet_dualenc_after_scratch_queue.pid"
LOG="${VIS}/nohup_nnunet_mim_dpdnet_dualenc_after_scratch_queue.log"

mkdir -p "${VIS}"
if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" && "${old}" != "$$" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[mim-dualenc-queue] already running pid=${old}"
    exit 0
  fi
fi
echo $$ > "${PID_FILE}"
exec > >(tee -a "${LOG}") 2>&1

echo "[mim-dualenc-queue] $(date '+%F %T') start pid=$$"

if [[ -f "${DONE_SELF}" ]] && grep -q 'status=ok' "${DONE_SELF}"; then
  echo "[mim-dualenc-queue] already done (marker)"
  exit 0
fi

_pgrep_real() {
  local pat="$1"
  pgrep -af "$pat" 2>/dev/null | grep -Ev 'queue_keeper|pgrep|run_nnunet_mim_dpdnet_dualenc_after_scratch' | grep -q .
}

echo "[mim-dualenc-queue] waiting MONAI scratch + SegAnyPET scratch 9fold DONE"
while true; do
  if [[ -f "${DONE_MONAI}" ]] && grep -q 'status=ok' "${DONE_MONAI}" \
    && [[ -f "${DONE_SEG}" ]] && grep -q 'status=ok' "${DONE_SEG}"; then
    if ! _pgrep_real 'run_aligned_monai_scratch_9fold_pipeline' \
      && ! _pgrep_real 'run_aligned_seganypet_scratch_9fold_pipeline'; then
      echo "[mim-dualenc-queue] both scratch pipelines done"
      break
    fi
  fi
  echo "[mim-dualenc-queue] waiting scratch… $(TZ=Asia/Shanghai date +%H:%M:%S)"
  sleep "${POLL_SEC}"
done

echo "[mim-dualenc-queue] launch nnUNet MIM FDG→PSMA fs50"
bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
bash "${ROOT}/ICLR2026/run/run_nnunet_mim_aligned_fdg_psma_bg.sh"

echo "[mim-dualenc-queue] launch DpDNet dual-enc FDG→PSMA fs50"
if _pgrep_real 'run_dpdnet_dualenc_aligned_fdg_psma|run_dpdnet_fdg_1gpu_bs6|iclr2026_dpdnet_dualenc'; then
  echo "[mim-dualenc-queue] DpDNet dual-enc already running — skip duplicate"
else
  bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
  bash "${ROOT}/ICLR2026/run/run_dpdnet_dualenc_aligned_fdg_psma_bg.sh"
fi

{
  echo "done_at=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "status=ok"
} > "${DONE_SELF}"

echo "[mim-dualenc-queue] ALL DONE $(date '+%F %T')"
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json" || true
