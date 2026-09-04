#!/usr/bin/env bash
set -euo pipefail
export TASK1_BASE=/media/ybwang/data1/PSMA-DATA
export TASK1_CONTINUE_TRAINING=0
export TASK1_CONTINUE_PICK_NEWER=0
export TASK1_CONTINUE_FROM_LATEST=0
export TASK1_CONTINUE_FROM_BEST=0
unset TASK1_NNUNET_RESULTS_STAMP_NAME || true
unset TASK1_GUARD_STAMP || true
bash /media/ybwang/data1/PSMA-CTRL/ICLR2026/run/run_aligned_nnunet_then_dpdnet_mae_fullcase_bg.sh
