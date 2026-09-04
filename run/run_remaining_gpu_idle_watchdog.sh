#!/usr/bin/env bash
# Self-check: if GPUs 0,1,3 stay idle while remaining schedule is supposed to run,
# kick the next pending 1-GPU jobs. Does not restart dead 9fold queues.
#
#   bash ICLR2026/run/run_remaining_gpu_idle_watchdog.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
HOLD="${VIS}/TASK1_REMAINING_SCHEDULE_HOLD.txt"
LOG="${VIS}/nohup_remaining_gpu_idle_watchdog.log"
PID_FILE="${VIS}/remaining_gpu_idle_watchdog.pid"
IDLE_NEED="${TASK1_REMAIN_WATCH_IDLE_SEC:-90}"
POLL="${TASK1_REMAIN_WATCH_POLL_SEC:-20}"

if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" && "${old}" != "$$" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[remain-watch] already running pid=${old}"
    exit 0
  fi
fi
echo $$ > "${PID_FILE}"
exec >>"${LOG}" 2>&1
echo "[remain-watch] start $(date '+%F %T') pid=$$ idle=${IDLE_NEED}s"

idle_since=""
_gpu_our_busy() {
  python3 - <<'PY'
import subprocess
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
)
busy = 0
for line in out.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        continue
    idx, mem, util = int(parts[0]), float(parts[1]), float(parts[2])
    if idx in (0, 1, 3) and (mem >= 400 or util >= 8):
        busy += 1
print(busy)
PY
}

_has_our_docker() {
  docker ps --format '{{.Command}} {{.Names}}' 2>/dev/null | grep -Eiq 'nnUNetv2_|dpdnet_|seganypet|mae_|cuda'
}

_kick_1gpu() {
  if pgrep -f 'run_nnunet_psma_fc70_decline_and_test_bg.sh|run_eval_fdg_shared_test20_bg.sh|run_eval_fdg_test20_bg.sh|run_dpdnet_dualenc_retrain_encoders' >/dev/null; then
    echo "[remain-watch] $(date '+%T') kick skipped — 1GPU/encoder already running"
    return 0
  fi
  echo "[remain-watch] $(date '+%T') GPUs idle — kick MIM 1GPU×3"
  python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD}" --no-plot \
    --patch-json '{"updated_note":"watchdog · idle kick 1GPU×3"}' || true
  TASK1_BOARD_METHOD=nnunet_mim \
    TASK1_UDA_FDG_STAMP=20260829_133121_iclr2026_nnunet_mim_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep \
    TASK1_UDA_FDG_BEST=/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/20260829_133121_iclr2026_nnunet_mim_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth \
    TASK1_PSMA_FC70_GPU=0 \
    nohup bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fc70_decline_and_test_bg.sh" \
      >>"${VIS}/nohup_watchdog_mim_fc70.log" 2>&1 &
  METHOD=nnunet_mim TASK1_TEST_SKIP_DONE=0 TASK1_CUDA_VISIBLE_DEVICES=1 \
    nohup bash "${ROOT}/ICLR2026/run/run_eval_fdg_shared_test20_bg.sh" \
      >>"${VIS}/nohup_watchdog_mim_fs0.log" 2>&1 &
  METHOD=nnunet_mim TASK1_TEST_SKIP_DONE=0 TASK1_CUDA_VISIBLE_DEVICES=3 \
    nohup bash "${ROOT}/ICLR2026/run/run_eval_fdg_test20_bg.sh" \
      >>"${VIS}/nohup_watchdog_mim_fdg.log" 2>&1 &
}

while [[ -f "${HOLD}" ]]; do
  busy="$(_gpu_our_busy || echo 1)"
  if [[ "${busy}" -gt 0 ]] || _has_our_docker; then
    idle_since=""
  else
    now="$(date +%s)"
    if [[ -z "${idle_since}" ]]; then
      idle_since="${now}"
      echo "[remain-watch] $(date '+%T') GPUs 0/1/3 idle start"
    fi
    elapsed=$((now - idle_since))
    if [[ "${elapsed}" -ge "${IDLE_NEED}" ]]; then
      _kick_1gpu
      idle_since="${now}"
    fi
  fi
  sleep "${POLL}"
done
echo "[remain-watch] HOLD gone — exit $(date '+%F %T')"
rm -f "${PID_FILE}"
