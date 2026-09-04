#!/usr/bin/env bash
# nnUNet MIM only: fill extra folds 0,1,3,4,6,7 on GPU1 + GPU3 (not GPU0/2).
#   bash ICLR2026/run/run_nnunet_mim_extra_folds_gpu13_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
LOOP="${ROOT}/ICLR2026/run/run_nnunet_mim_extra_fold_loop.sh"
mkdir -p "${VIS}"
chmod +x "${LOOP}" "${ROOT}/ICLR2026/run/run_nnunet_mim_extra_fold_onegpu.sh"

for gpu in 1 3; do
  pidf="${VIS}/nnunet_mim_extra_fold_gpu${gpu}.pid"
  logf="${VIS}/nohup_nnunet_mim_extra_fold_gpu${gpu}.log"
  if [[ -f "${pidf}" ]]; then
    old="$(tr -d '[:space:]' < "${pidf}" || true)"
    if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
      echo "[mim-extra] gpu${gpu} already running pid=${old}"
      continue
    fi
  fi
  nohup bash "${LOOP}" --gpu "${gpu}" >>"${logf}" 2>&1 &
  echo $! > "${pidf}"
  echo "[mim-extra] started gpu${gpu} pid=$(cat "${pidf}") log=${logf}"
done
