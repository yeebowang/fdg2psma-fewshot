#!/usr/bin/env bash
# CPU: build 1ch FDG raw + nnUNet plan/preprocess for dual-enc retrain (no GPU wait).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
LOG="${VIS}/nohup_dpdnet_dualenc_encoder_cpu_prep.log"
PID_FILE="${VIS}/dpdnet_dualenc_encoder_cpu_prep.pid"
WORKER="${ROOT}/ICLR2026/run/run_dpdnet_dualenc_encoder_cpu_prep_worker.sh"

mkdir -p "${VIS}"
if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[enc-prep] already running pid=${old}"
    exit 0
  fi
fi
chmod +x "${WORKER}"
nohup bash "${WORKER}" >>"${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "[enc-prep] started pid=$(cat "${PID_FILE}") log=${LOG}"
