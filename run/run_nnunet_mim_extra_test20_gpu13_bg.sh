#!/usr/bin/env bash
# nnUNet MIM: fill missing TEST20 on extra folds using GPU1 + GPU3.
#   bash ICLR2026/run/run_nnunet_mim_extra_test20_gpu13_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
LOOP="${ROOT}/ICLR2026/run/run_nnunet_mim_extra_fold_test20_loop.sh"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
mkdir -p "${VIS}"
chmod +x "${LOOP}" "${ROOT}/ICLR2026/run/run_nnunet_mim_extra_fold_test20_onegpu.sh"

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json '{"methods":{"nnunet_mim":{"psma_fs50_f258":{"note":"train 9/9 · TEST20 filling"},"psma_fs10_f258":{"note":"train 9/9 · TEST20 filling"},"psma_fs5_f258":{"note":"train 9/9 · TEST20 filling"}}},"updated_note":"nnUNet MIM train 9/9 done · start TEST20 on GPU1+3"}' || true

for gpu in 1 3; do
  pidf="${VIS}/nnunet_mim_extra_test20_gpu${gpu}.pid"
  logf="${VIS}/nohup_nnunet_mim_extra_test20_gpu${gpu}.log"
  if [[ -f "${pidf}" ]]; then
    old="$(tr -d '[:space:]' < "${pidf}" || true)"
    if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
      echo "[mim-test20] gpu${gpu} already running pid=${old}"
      continue
    fi
  fi
  nohup bash "${LOOP}" --gpu "${gpu}" >>"${logf}" 2>&1 &
  echo $! > "${pidf}"
  echo "[mim-test20] started gpu${gpu} pid=$(cat "${pidf}") log=${logf}"
done
