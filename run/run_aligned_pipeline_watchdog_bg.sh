#!/usr/bin/env bash
# Watchdog for aligned FDG→PSMA pipeline: detect idle stall and resume from next stage.
#
# Does NOT kill healthy training. Only acts when:
#   - no stage runner / resume process
#   - GPUs 0/1/3 look idle (low mem)
#   - board/filesystem say work remains
#
#   bash ICLR2026/run/run_aligned_pipeline_watchdog_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"
STATE_JSON="${LOG_DIR}/aligned_pipeline_state.json"
NN_WORK="${DATA}/task1_train_workspace/nnUNet_results"
WATCH_LOG="${LOG_DIR}/nohup_aligned_pipeline_watchdog.log"
INTERVAL_SEC="${TASK1_ALIGN_WATCHDOG_SEC:-120}"
IDLE_MEM_MIB="${TASK1_ALIGN_IDLE_MEM_MIB:-800}"   # below this on 0/1/3 → idle
STALL_SEC="${TASK1_ALIGN_STALL_SEC:-900}"         # 15min idle+pending → resume
MAE_STAMP="${TASK1_MAE_WAIT_STAMP:-20260816_161859_iclr2026_mae_psma_fs50_from_fdg_seg_f258_gpu013}"

mkdir -p "${LOG_DIR}"
exec >>"${WATCH_LOG}" 2>&1
echo "[watchdog] start $(TZ=Asia/Shanghai date '+%F %T') interval=${INTERVAL_SEC}s stall=${STALL_SEC}s"

_pipeline_alive() {
  pgrep -f 'run_aligned_pipeline_stages_bg.sh|run_aligned_resume_after_mae_test20_bg.sh|run_aligned_continue_after_mae_nnunet_first_bg.sh' >/dev/null 2>&1
}

_nnunet_train_alive() {
  # parent wait script or fold containers/guards
  pgrep -f 'run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh' >/dev/null 2>&1 && return 0
  docker ps --format '{{.Image}}' 2>/dev/null | grep -q 'autopet_baseline' && return 0
  return 1
}

_gpus_idle() {
  # return 0 if GPUs 0,1,3 all below IDLE_MEM_MIB
  python3 - <<PY
import subprocess
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
idle = True
for line in out.strip().splitlines():
    idx, used = [x.strip() for x in line.split(",")]
    if idx in ("0", "1", "3") and float(used) >= float("${IDLE_MEM_MIB}"):
        idle = False
print("1" if idle else "0")
PY
}

_decide_next_stage() {
  BOARD_JSON="${BOARD_JSON}" REPO="${REPO}" LOG_DIR="${LOG_DIR}" NN_WORK="${NN_WORK}" MAE_STAMP="${MAE_STAMP}" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

board_p = Path(os.environ["BOARD_JSON"])
repo = Path(os.environ["REPO"]) / "runs"
mae = os.environ["MAE_STAMP"]
log_dir = Path(os.environ["LOG_DIR"])
nn_work = Path(os.environ["NN_WORK"])

board = json.loads(board_p.read_text()) if board_p.is_file() else {}
methods = board.get("methods") or {}

def st(m, s):
    return (methods.get(m) or {}).get(s) or {}

mae_fs = st("mae_swinunetr", "psma_fs50_f258")
nn = st("nnunet", "psma_fs50_f258")
monai_fdg = st("monai_swinvit", "fdg_pretrain")
monai_fs = st("monai_swinvit", "psma_fs50_f258")
seg_fdg = st("seganypet", "fdg_pretrain")
seg_fs = st("seganypet", "psma_fs50_f258")

mae_test = repo / mae / "psma_test20_eval" / "aggregate_test20_f258.json"
mae_done = mae_test.is_file() or (
    mae_fs.get("status") == "done" and mae_fs.get("mean") is not None
)

nn_stamp = (nn.get("stamp") or "").strip()
nn_done = bool(nn.get("status") == "done" and nn.get("mean") is not None)
if nn_stamp and (log_dir / f"aggregate_nnunet_psma_fs50_f258_{nn_stamp}.json").is_file():
    nn_done = True
if not nn_done:
    for _p in sorted(log_dir.glob("aggregate_nnunet_psma_fs50_f258_*val70e20*.json"), reverse=True):
        nn_done = True
        break

nn_running = False
if nn_stamp:
    finals = []
    for f in (2, 5, 8):
        fold = (
            nn_work
            / f"{nn_stamp}_f{f}"
            / "Dataset228_AutoPETIV_Task1_2ch"
            / "nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres"
            / "fold_0"
        )
        live = fold / "task1_train_live_progress.json"
        final = fold / "checkpoint_final.pth"
        finals.append(final.is_file())
        if live.is_file() and not final.is_file():
            nn_running = True
    if nn.get("status") == "running" and not all(finals) and any(
        (nn_work / f"{nn_stamp}_f{f}").exists() for f in (2, 5, 8)
    ):
        nn_running = True

monai_done = bool(monai_fs.get("status") == "done" and monai_fs.get("mean") is not None)
if not monai_done:
    for p in sorted(repo.glob("*monai*/psma_test20_eval/aggregate_test20_f258.json"), reverse=True):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("method") == "monai" and d.get("test_mean") is not None:
            monai_done = True
            break

seg_done = seg_fs.get("status") == "done"
monai_running = monai_fdg.get("status") == "running" or monai_fs.get("status") == "running"
seg_running = seg_fdg.get("status") == "running" or seg_fs.get("status") == "running"

if nn_running:
    print("WAIT_NNUNET")
elif monai_running:
    print("WAIT_MONAI")
elif seg_running:
    print("WAIT_SEGANY")
elif not mae_done:
    print("WAIT_MAE")
elif not nn_done:
    print("START_NNUNET")
elif not monai_done:
    print("START_MONAI")
elif not seg_done:
    print("START_SEGANY")
else:
    print("ALL_DONE")
PY
}

idle_since=""
while true; do
  action="$(_decide_next_stage)"
  alive=0
  _pipeline_alive && alive=1
  nn_alive=0
  _nnunet_train_alive && nn_alive=1
  idle="$(_gpus_idle)"
  now=$(date +%s)
  echo "[watchdog] $(TZ=Asia/Shanghai date '+%F %T') action=${action} pipeline_alive=${alive} nn_alive=${nn_alive} gpus_idle=${idle}"

  if [[ "${action}" == "ALL_DONE" ]]; then
    echo "[watchdog] all stages done; exiting"
    break
  fi

  # healthy: pipeline or nnunet training running
  if [[ "${alive}" -eq 1 ]] || [[ "${nn_alive}" -eq 1 ]]; then
    idle_since=""
    # refresh heartbeat observation
    python3 - <<PY
import json, time
from pathlib import Path
p = Path("${STATE_JSON}")
d = json.loads(p.read_text()) if p.is_file() else {}
d["watchdog_seen_unix"] = time.time()
d["last_action"] = "${action}"
d["pipeline_alive"] = bool(${alive})
d["nn_alive"] = bool(${nn_alive})
p.write_text(json.dumps(d, indent=2) + "\n")
PY
    sleep "${INTERVAL_SEC}"
    continue
  fi

  # stalled candidate: need to start something and GPUs idle
  if [[ "${action}" == START_* ]] && [[ "${idle}" == "1" ]]; then
    if [[ -z "${idle_since}" ]]; then
      idle_since="${now}"
      echo "[watchdog] idle+pending since ${idle_since} action=${action}"
    fi
    waited=$((now - idle_since))
    if [[ "${waited}" -ge "${STALL_SEC}" ]]; then
      echo "[watchdog] STALL ${waited}s → resume ${action}"
      case "${action}" in
        START_NNUNET)
          nohup env FROM_STAGE=nnunet bash "${CTRL}/ICLR2026/run/run_aligned_pipeline_stages_bg.sh" \
            >/dev/null 2>&1 &
          echo "[watchdog] launched FROM_STAGE=nnunet pid=$!"
          ;;
        START_MONAI)
          nohup env FROM_STAGE=monai bash "${CTRL}/ICLR2026/run/run_aligned_pipeline_stages_bg.sh" \
            >/dev/null 2>&1 &
          echo "[watchdog] launched FROM_STAGE=monai pid=$!"
          ;;
        START_SEGANY)
          nohup env FROM_STAGE=seganypet bash "${CTRL}/ICLR2026/run/run_aligned_pipeline_stages_bg.sh" \
            >/dev/null 2>&1 &
          echo "[watchdog] launched FROM_STAGE=seganypet pid=$!"
          ;;
      esac
      idle_since=""
      sleep 30
    fi
  else
    idle_since=""
  fi
  sleep "${INTERVAL_SEC}"
done
