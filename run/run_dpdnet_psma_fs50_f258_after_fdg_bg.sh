#!/usr/bin/env bash
# Wait for DpDNet FDG to finish, then launch PSMA fs50 f258 (tr25/val25e20/100ep/bs2).
# FDG→PSMA init: checkpoint_final > checkpoint_latest（不用 val-best）。
#   bash ICLR2026/run/run_dpdnet_psma_fs50_f258_after_fdg_bg.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ICLR_VIS="${ROOT}/ICLR2026/vis"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
LOG="${ICLR_VIS}/nohup_dpdnet_psma_after_fdg_wait.log"

FDG_STAMP="${TASK1_DPDNET_FDG_STAMP:-}"
if [[ -z "${FDG_STAMP}" && -f "${ICLR_VIS}/dpdnet_fdg_LAST_STAMP.txt" ]]; then
  FDG_STAMP="$(tr -d '[:space:]' < "${ICLR_VIS}/dpdnet_fdg_LAST_STAMP.txt")"
fi
[[ -n "${FDG_STAMP}" ]] || { echo "[error] no FDG stamp" >&2; exit 1; }

FDG_DS="Dataset239_DpDNet_FDG_2ch"
FDG_TF="STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres"
FDG_FOLD="${WORK}/nnUNet_results/${FDG_STAMP}/${FDG_DS}/${FDG_TF}/fold_0"
FDG_FINAL="${FDG_FOLD}/checkpoint_final.pth"
FDG_LATEST="${FDG_FOLD}/checkpoint_latest.pth"
TOTAL_FDG_EP="${TASK1_DPDNET_FDG_WAIT_EPOCHS:-169}"

_resolve_fdg_ckpt() {
  local f
  for f in checkpoint_final.pth checkpoint_latest.pth; do
    if [[ -f "${FDG_FOLD}/${f}" ]]; then
      echo "${FDG_FOLD}/${f}"
      return 0
    fi
  done
  return 1
}

echo "[wait] FDG_STAMP=${FDG_STAMP} target_ep=${TOTAL_FDG_EP} init=final>latest" | tee "${LOG}"

# stop stale waiter of this stamp if we are relaunching from outside
if [[ -f "${ICLR_VIS}/dpdnet_psma_after_fdg_wait.pid" ]]; then
  old="$(tr -d '[:space:]' < "${ICLR_VIS}/dpdnet_psma_after_fdg_wait.pid" || true)"
  if [[ -n "${old}" && "${old}" != "$$" ]] && kill -0 "${old}" 2>/dev/null; then
    # only kill if it's the after-fdg waiter (not this shell)
    :
  fi
fi

nohup bash -lc "
set -euo pipefail
FDG_FOLD='${FDG_FOLD}'
FDG_FINAL='${FDG_FINAL}'
FDG_LATEST='${FDG_LATEST}'
FDG_STAMP='${FDG_STAMP}'
LOG='${LOG}'
ROOT='${ROOT}'
while true; do
  if [[ -f \"\${FDG_FINAL}\" ]]; then
    echo '[wait] FDG checkpoint_final present' | tee -a \"\${LOG}\"
    break
  fi
  cname=\"dpdnet_fdg_\${FDG_STAMP}\"
  if ! docker ps --format '{{.Names}}' | grep -qx \"\${cname}\"; then
    if [[ -f \"\${FDG_FINAL}\" || -f \"\${FDG_LATEST}\" ]]; then
      echo '[wait] FDG container gone + final/latest exists' | tee -a \"\${LOG}\"
      break
    fi
  fi
  echo \"[wait] \$(date '+%F %T') still waiting for FDG \${FDG_STAMP}…\" | tee -a \"\${LOG}\"
  sleep 120
done
FDG_CKPT=''
for f in checkpoint_final.pth checkpoint_latest.pth; do
  if [[ -f \"\${FDG_FOLD}/\${f}\" ]]; then
    FDG_CKPT=\"\${FDG_FOLD}/\${f}\"
    break
  fi
done
[[ -n \"\${FDG_CKPT}\" ]] || { echo '[error] no FDG final/latest ckpt' | tee -a \"\${LOG}\"; exit 1; }
echo \"[wait] FDG→PSMA init=\${FDG_CKPT}\" | tee -a \"\${LOG}\"
export TASK1_DPDNET_FDG_STAMP=\"\${FDG_STAMP}\"
export TASK1_DPDNET_FDG_BEST=\"\${FDG_CKPT}\"
export TASK1_DPDNET_SKIP_PREPARE=1
export TASK1_DPDNET_NUM_EPOCHS=100
export TASK1_DPDNET_TRAIN_ITERS=25
export TASK1_DPDNET_VAL_ITERS=25
export TASK1_DPDNET_VAL_EVERY=20
export TASK1_VAL_EVERY_N_EPOCHS=20
export TASK1_BEST_BY=val_loss
export TASK1_DPDNET_BATCH_SIZE=2
export TASK1_FOLDS=2,5,8
export TASK1_FOLD_GPUS=2:0,5:1,8:3
bash \"\${ROOT}/ICLR2026/run/run_dpdnet_psma_fewshot50_f258_1gpu_bs2_100ep_bg.sh\"
" >>"${LOG}" 2>&1 &
echo $! > "${ICLR_VIS}/dpdnet_psma_after_fdg_wait.pid"
echo "[wait] launched pid=$(cat "${ICLR_VIS}/dpdnet_psma_after_fdg_wait.pid") log=${LOG}"
