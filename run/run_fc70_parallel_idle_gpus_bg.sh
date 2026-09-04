#!/usr/bin/env bash
# Use idle GPUs (default 1,3) to advance queued work while fc70 pipeline holds GPU 0.
#
#   bash ICLR2026/run/run_fc70_parallel_idle_gpus_bg.sh
#   IDLE_GPUS=1,3 bash ICLR2026/run/run_fc70_parallel_idle_gpus_bg.sh
#
# Launches (when pending):
#   GPU1 → DpDNet PSMA fc70% train+TEST20
#   GPU3 → nnUNet FDG TEST20 rerun (failed 0/202 aggregate cleared)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
IDLE_GPUS="${IDLE_GPUS:-1,3}"
GPU_DPD="${GPU_DPD:-1}"
GPU_NN="${GPU_NN:-3}"

_fc70_pending() {
  local mkey="$1"
  python3 - <<PY
import json, sys
from pathlib import Path
b = json.loads(Path("${BOARD}").read_text())
st = (b.get("methods") or {}).get("${mkey}", {}).get("psma_fc70") or {}
s = (st.get("status") or "pending").lower()
sys.exit(0 if s in ("pending", "queued", "") else 1)
PY
}

_fdg_test_nnunet_pending() {
  python3 - <<PY
import json, math, sys
from pathlib import Path
agg = Path("${VIS}/fdg_test20/aggregate_nnunet.json")
if not agg.is_file():
    sys.exit(0)
try:
    d = json.loads(agg.read_text())
except Exception:
    sys.exit(0)
md = d.get("mean_dice", d.get("mean"))
if isinstance(md, (int, float)) and md == md:
    sys.exit(1)
n = d.get("n_scored")
if isinstance(n, (int, float)) and int(n) > 0:
    sys.exit(1)
sys.exit(0)
PY
}

_gpu_free() {
  local gid="$1"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v g="${gid}" '$1==g { if ($2+0 < 15 && $3+0 < 2048) exit 0; exit 1 }'
}

echo "[idle-gpu] IDLE_GPUS=${IDLE_GPUS} dpdnet→gpu${GPU_DPD} nnunet-fdg-test→gpu${GPU_NN}"

_launched=0

if _fc70_pending dpdnet && _gpu_free "${GPU_DPD}"; then
  if pgrep -af 'run_dpdnet_psma_fc70_decline_and_test_bg.sh|dpdnet_psma_fc70_' 2>/dev/null \
      | grep -Ev 'pgrep|cursor|idle-gpu|parallel_idle' | grep -q .; then
    echo "[idle-gpu] skip dpdnet fc70 (already running)"
  else
    LOG="${VIS}/nohup_dpdnet_psma_fc70_gpu${GPU_DPD}_parallel.log"
    echo "[idle-gpu] launch DpDNet PSMA fc70% on GPU ${GPU_DPD} → ${LOG}"
    nohup env TASK1_PSMA_FC70_GPU="${GPU_DPD}" \
      bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_fc70_decline_and_test_bg.sh" \
      >>"${LOG}" 2>&1 &
    echo $! > "${VIS}/dpdnet_psma_fc70_gpu${GPU_DPD}.pid"
    _launched=$((_launched + 1))
  fi
else
  echo "[idle-gpu] skip dpdnet fc70 (not pending or GPU ${GPU_DPD} busy)"
fi

if _fdg_test_nnunet_pending && _gpu_free "${GPU_NN}"; then
  if pgrep -af 'run_eval_fdg_test20_bg.sh|fdg_test20_eval/nnunet' 2>/dev/null | grep -Ev 'pgrep|idle-gpu' | grep -q .; then
    echo "[idle-gpu] skip nnUNet FDG TEST (already running)"
  else
    rm -f "${VIS}/fdg_test20/aggregate_nnunet.json"
    LOG="${VIS}/nohup_fdg_test20_nnunet_gpu${GPU_NN}_rerun.log"
    echo "[idle-gpu] launch nnUNet FDG TEST on GPU ${GPU_NN} → ${LOG}"
    nohup env METHOD=nnunet TASK1_TEST_SKIP_DONE=0 \
      TASK1_CUDA_VISIBLE_DEVICES="${GPU_NN}" TASK1_UDA_PRED_PER_GPU=1 \
      bash "${ROOT}/ICLR2026/run/run_eval_fdg_test20_bg.sh" \
      >>"${LOG}" 2>&1 &
    echo $! > "${VIS}/fdg_test20_nnunet_gpu${GPU_NN}.pid"
    _launched=$((_launched + 1))
  fi
else
  echo "[idle-gpu] skip nnUNet FDG TEST (done or GPU ${GPU_NN} busy)"
fi

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"updated_note\":\"idle GPU parallel launch (${IDLE_GPUS}) · started=${_launched}\"}" || true

echo "[idle-gpu] launched=${_launched} (MONAI fc70 continues on GPU 0)"
