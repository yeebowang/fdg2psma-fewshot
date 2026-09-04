#!/usr/bin/env bash
# Sequential: nnUNet MAE-fullcase → DpDNet MAE-fullcase (GPUs 0,1,3).
#   bash ICLR2026/run/run_aligned_nnunet_then_dpdnet_mae_fullcase_bg.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ICLR_VIS="${ROOT}/ICLR2026/vis"
mkdir -p "${ICLR_VIS}"
LOG="${ICLR_VIS}/nohup_aligned_nnunet_then_dpdnet_mae_fullcase.log"

nohup bash -lc "
set -euo pipefail
ROOT='${ROOT}'
echo \"[queue] \$(date '+%F %T') start nnUNet MAE-fullcase\"
bash \"\${ROOT}/ICLR2026/run/run_nnunet_aligned_fdg_psma_f258_mae_fullcase_bg.sh\"
echo \"[queue] \$(date '+%F %T') nnUNet done → DpDNet MAE-fullcase\"
bash \"\${ROOT}/ICLR2026/run/run_dpdnet_aligned_fdg_psma_f258_mae_fullcase_bg.sh\"
echo \"[queue] \$(date '+%F %T') ALL DONE\"
" >>"${LOG}" 2>&1 &
echo $! > "${ICLR_VIS}/aligned_nnunet_then_dpdnet_mae_fullcase.pid"
echo "[queue] pid=$(cat "${ICLR_VIS}/aligned_nnunet_then_dpdnet_mae_fullcase.pid") log=${LOG}"
