#!/usr/bin/env bash
# SegAnyPET scratch · same protocol as SegAnyPET row, FDG init = random (no lesion).
# FDG 3GPU bs=6 100ep → PSMA fs50/10/5 9-fold (waves of 3) → TEST20 → fc70 → fs0 → FDG TEST.
#
#   bash ICLR2026/run/run_aligned_seganypet_scratch_9fold_pipeline_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
FOLDS_CSV="${TASK1_SEGANY_SCRATCH_FOLDS:-0,1,2,3,4,5,6,7,8}"
GPUS="${TASK1_CUDA_VISIBLE_DEVICES:-0,1,3}"
POLL="${TASK1_CHAIN_POLL_SEC:-30}"
IDLE_MIB="${TASK1_GPU_IDLE_MEM_MIB:-2048}"
PID_FILE="${VIS}/seganypet_scratch_9fold_pipeline.pid"
LOG="${VIS}/nohup_seganypet_scratch_9fold_pipeline.log"
LAST_FDG="${VIS}/seganypet_scratch_fdg_LAST_STAMP.txt"
DONE_MARK="${VIS}/TASK1_SEGANY_SCRATCH_9FOLD_DONE.txt"

mkdir -p "${VIS}"
if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" && "${old}" != "$$" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[seganypet-scratch] already running pid=${old}"
    exit 0
  fi
fi
echo $$ > "${PID_FILE}"
exec > >(tee -a "${LOG}") 2>&1

echo "[seganypet-scratch] $(date '+%F %T') start pid=$$ folds=${FOLDS_CSV} gpus=${GPUS}"

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD}"
export TASK1_BOARD_METHOD=seganypet_scratch
export TASK1_SEGANY_CKPT=none
export TASK1_CUDA_VISIBLE_DEVICES="${GPUS}"
export TASK1_DOCKER_GPUS="device=${GPUS}"
export TASK1_PREFLIGHT_GPUS="${GPUS//,/ }"
export TASK1_SEGANY_GPU_LIST="${GPUS//,/ }"
export TASK1_SEGANY_OFFICIAL_GPUS="${GPUS}"
export TASK1_SEGANY_FOLDS_CSV="${FOLDS_CSV}"

bash "${CTRL}/run_task/run_task1_train_auto_resume_guard_bg.sh" || true

_disarm() { bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true; }
_arm() { bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true; }

_board() {
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD}" --no-plot --patch-json "$1" || true
}

_board_get() {
  python3 - <<PY
import json
from pathlib import Path
b = json.loads(Path("${BOARD}").read_text()) if Path("${BOARD}").is_file() else {}
m = (b.get("methods") or {}).get("seganypet_scratch") or {}
st = m.get("${1}") or {}
v = st.get("${2}")
if v is None:
    print("")
else:
    print(str(v).strip())
PY
}

_gpus_idle() {
  python3 - <<PY
import subprocess, sys
gpus = [x.strip() for x in "${GPUS}".split(",") if x.strip()]
lim = int("${IDLE_MIB}")
try:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15,
    )
except Exception:
    raise SystemExit(1)
used = {}
for line in (r.stdout or "").splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2:
        used[str(int(float(parts[0])))] = int(float(parts[1]))
for g in gpus:
    if used.get(g, 10**9) >= lim:
        print(f"busy gpu={g} mem={used.get(g)}MiB")
        raise SystemExit(1)
print("idle")
PY
}

_wait_gpus_idle() {
  echo "[seganypet-scratch] wait GPUs ${GPUS} idle (<${IDLE_MIB} MiB)"
  while ! _gpus_idle >/dev/null; do
    echo "[seganypet-scratch] GPUs busy… $(TZ=Asia/Shanghai date +%H:%M:%S)"
    sleep "${POLL}"
  done
  sleep 15
  while ! _gpus_idle >/dev/null; do
    echo "[seganypet-scratch] GPUs busy again… $(TZ=Asia/Shanghai date +%H:%M:%S)"
    sleep "${POLL}"
  done
  echo "[seganypet-scratch] GPUs idle"
}

_remaining_folds() {
  local stamp="$1"
  FOLDS_CSV="${FOLDS_CSV}" STAMP="${stamp}" REPO="${REPO}" python3 - <<'PY'
import os
from pathlib import Path
stamp = os.environ["STAMP"].strip()
folds = [x.strip() for x in os.environ["FOLDS_CSV"].split(",") if x.strip()]
repo = Path(os.environ["REPO"]) / "runs" / stamp / "seganypet"
out = []
for f in folds:
    d = repo / f"fold{f}"
    if (d / "best.pth").is_file() or (d / "latest.pth").is_file():
        continue
    out.append(f)
print(",".join(out))
PY
}

# ---------- FDG scratch ----------
FDG_STAMP="$(_board_get fdg_pretrain stamp)"
if [[ -z "${FDG_STAMP}" && -f "${LAST_FDG}" ]]; then
  FDG_STAMP="$(tr -d '[:space:]' < "${LAST_FDG}")"
fi
if [[ -z "${FDG_STAMP}" ]]; then
  FDG_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_scratch_fdg_pretrain_gpu013"
fi
FDG_BEST="${REPO}/runs/${FDG_STAMP}/seganypet_fdg/best.pth"
FDG_LATEST="${REPO}/runs/${FDG_STAMP}/seganypet_fdg/latest.pth"
CNAME="seganypet_fdg_${FDG_STAMP}"

if [[ ! -f "${FDG_BEST}" && ! -f "${FDG_LATEST}" ]]; then
  _wait_gpus_idle
  _disarm
  _board "{\"methods\":{\"seganypet_scratch\":{\"fdg_pretrain\":{\"status\":\"running\",\"stamp\":\"${FDG_STAMP}\",\"bs\":6,\"bs_note\":\"global DP 0,1,3\",\"total_epochs\":100,\"note\":\"scratch → FDG 100ep\"}}},\"updated_note\":\"SegAnyPET scratch FDG running\"}"
  export TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}"
  export TASK1_BOARD_METHOD=seganypet_scratch
  export TASK1_SEGANY_CKPT=none
  bash "${CTRL}/ICLR2026/run/run_seganypet_fdg_pretrain_bg.sh"
  echo "[seganypet-scratch] waiting FDG container ${CNAME}"
  for _i in $(seq 1 180); do
    if docker ps --format '{{.Names}}' | grep -qx "${CNAME}"; then
      break
    fi
    if [[ -f "${FDG_BEST}" || -f "${FDG_LATEST}" ]]; then
      break
    fi
    sleep 5
  done
  while docker ps --format '{{.Names}}' | grep -qx "${CNAME}"; do
    echo "[seganypet-scratch] FDG still running $(TZ=Asia/Shanghai date +%H:%M:%S)"
    sleep "${POLL}"
  done
  _arm
fi
FOUNDATION=""
for cand in "${FDG_BEST}" "${FDG_LATEST}"; do
  [[ -f "${cand}" ]] && { FOUNDATION="${cand}"; break; }
done
[[ -n "${FOUNDATION}" ]] || { echo "[error] no FDG scratch ckpt ${FDG_STAMP}" >&2; exit 1; }
echo "${FDG_STAMP}" > "${LAST_FDG}"
_board "{\"methods\":{\"seganypet_scratch\":{\"fdg_pretrain\":{\"status\":\"done\",\"stamp\":\"${FDG_STAMP}\",\"best_ckpt\":\"${FOUNDATION}\",\"bs\":6,\"bs_note\":\"global DP 0,1,3\",\"total_epochs\":100,\"note\":\"scratch → FDG 100ep DONE\"}}},\"updated_note\":\"SegAnyPET scratch FDG done\"}"
echo "[seganypet-scratch] FDG ckpt=${FOUNDATION}"

export TASK1_SEGANY_CKPT="${FOUNDATION}"

FOLD_GPUS_9="0:0,1:1,2:3,3:0,4:1,5:3,6:0,7:1,8:3"

_run_fewshot() {
  local n="$1"
  local stage="psma_fs${n}_f258"
  local stamp
  stamp="$(_board_get "${stage}" stamp)"
  if [[ -z "${stamp}" ]]; then
    stamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_scratch_psma_fs${n}_f258_gpu013"
  fi
  local remain
  remain="$(_remaining_folds "${stamp}")"
  if [[ -n "${remain}" ]]; then
    echo "[seganypet-scratch] fs${n} train folds=${remain} stamp=${stamp}"
    export TASK1_NNUNET_RESULTS_STAMP_NAME="${stamp}"
    _disarm
    _board "{\"methods\":{\"seganypet_scratch\":{\"${stage}\":{\"status\":\"running\",\"stamp\":\"${stamp}\",\"note\":\"9fold · fs${n} · remaining ${remain}\"}}},\"updated_note\":\"SegAnyPET scratch fs${n} 9fold\"}"
    TASK1_FEWSHOT_N="${n}" \
    TASK1_PSMA_BOARD_STAGE="${stage}" \
    TASK1_BOARD_METHOD=seganypet_scratch \
    TASK1_SEGANY_CKPT="${FOUNDATION}" \
    TASK1_SEGANY_FOLDS_CSV="${remain}" \
    TASK1_SEGANY_GPU_LIST="${GPUS//,/ }" \
    TASK1_NNUNET_RESULTS_STAMP_NAME="${stamp}" \
      bash "${CTRL}/ICLR2026/run/run_seganypet_fewshot50_f258_bg.sh"
    _arm
  else
    echo "[seganypet-scratch] fs${n} all folds present stamp=${stamp}"
  fi
  echo "[seganypet-scratch] fs${n} TEST20 9fold"
  _disarm
  STAMP="${stamp}" \
    TASK1_FEWSHOT_N="${n}" \
    TASK1_PSMA_BOARD_STAGE="${stage}" \
    TASK1_SEGANY_FOLDS_CSV="${FOLDS_CSV}" \
    TASK1_FOLD_GPUS="${FOLD_GPUS_9}" \
    TASK1_TEST_SKIP_DONE=1 \
      bash "${CTRL}/ICLR2026/run/run_eval_seganypet_psma_test20_f258_bg.sh"
}

_run_fewshot 50
_run_fewshot 10
_run_fewshot 5

# ---------- fc70 ----------
fc70_st="$(_board_get psma_fc70 status)"
fc70_mean="$(_board_get psma_fc70 mean)"
if pgrep -af 'run_seganypet_scratch_psma_fc70_from_fdg_bg.sh|iclr2026_seganypet_scratch_psma_fc70' 2>/dev/null \
    | grep -Ev 'pgrep|queue_keeper|gpu_idle_queue' | grep -q .; then
  echo "[seganypet-scratch] fc70 already running (idle/other) — skip duplicate launch"
elif [[ "${fc70_st}" != "done" || -z "${fc70_mean}" ]]; then
  FC70_STAMP="$(_board_get psma_fc70 stamp)"
  [[ -n "${FC70_STAMP}" ]] || FC70_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_scratch_psma_fc70_gpu0"
  echo "[seganypet-scratch] fc70 stamp=${FC70_STAMP}"
  _disarm
  TASK1_BOARD_METHOD=seganypet_scratch \
  TASK1_SEGANY_CKPT="${FOUNDATION}" \
  TASK1_NNUNET_RESULTS_STAMP_NAME="${FC70_STAMP}" \
  TASK1_PSMA_FC70_GPU=0 \
    bash "${CTRL}/ICLR2026/run/run_seganypet_scratch_psma_fc70_from_fdg_bg.sh"
  _arm
else
  echo "[seganypet-scratch] fc70 already done"
fi

# ---------- PSMA fs0 / FDG TEST ----------
_disarm
METHOD=seganypet_scratch TASK1_TEST_SKIP_DONE=0 TASK1_CUDA_VISIBLE_DEVICES=0 \
  bash "${CTRL}/ICLR2026/run/run_eval_fdg_shared_test20_bg.sh" || true
METHOD=seganypet_scratch TASK1_TEST_SKIP_DONE=0 TASK1_CUDA_VISIBLE_DEVICES=0 \
  bash "${CTRL}/ICLR2026/run/run_eval_fdg_test20_bg.sh" || true
_arm

{
  echo "done_at=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "status=ok"
  echo "fdg_stamp=${FDG_STAMP}"
} > "${DONE_MARK}"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true
echo "[seganypet-scratch] ALL DONE $(date '+%F %T')"
