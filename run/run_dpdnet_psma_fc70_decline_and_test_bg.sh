#!/usr/bin/env bash
# DpDNet PSMA fc70% · single run · tr25 val25 · decline → TEST20
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/ICLR2026/run/_psma_fc70_env.sh"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
MON="${ROOT}/ICLR2026/scripts/monitor_val_dice_decline_stop.py"

read -r TR VAL NTR NVAL < <(_fc70_resolve_iters)
DPD_FDG_BEST="${TASK1_DPDNET_FDG_BEST:-/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/20260817_165250_iclr2026_dpdnet_fdg_2gpu_bs3_gbs6_n6_tr70_val0_169ep_gpu01/Dataset239_DpDNet_FDG_2ch/STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth}"
BOARD_METHOD="${TASK1_BOARD_METHOD:-dpdnet}"

PARENT="${TASK1_NNUNET_RESULTS_STAMP_NAME:-}"
[[ -n "${PARENT}" && "${PARENT}" == *fc70* ]] || \
  PARENT="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_${BOARD_METHOD}_psma_fc70_1gpu_bs${PSMA_FC70_BS}_tr${TR}_val${VAL}e${PSMA_FC70_VAL_EVERY}_${PSMA_FC70_EP}ep_gpu${PSMA_FC70_GPU}"

PIPE_LOG="${VIS}/nohup_dpdnet_psma_fc70_${PARENT}.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1
echo "[dpdnet-fc70] PARENT=${PARENT} n=${NTR}/${NVAL} init=${DPD_FDG_BEST}"

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"${PSMA_FC70_STAGE}\":{\"status\":\"running\",\"stamp\":\"${PARENT}\",\"note\":\"fc70% single run\"}}},\"updated_note\":\"${BOARD_METHOD} fc70\"}" || true

bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true

# prepare Dataset240 with fc70 split once
if [[ "${TASK1_DPDNET_SKIP_PREPARE:-0}" != "1" ]]; then
  docker run --rm --user root -v "${ROOT}:${ROOT}" -v "${DATA}:${DATA}" --entrypoint python3 autopet_baseline:latest \
    "${ROOT}/ICLR2026/scripts/prepare_dpdnet_psma_dataset240.py" \
      --work "${WORK}" --dst-id 240 --batch-size "${PSMA_FC70_BS}" \
      --n-folds 1 --split-dir "${ROOT}/ICLR2026/data"
fi

FOLD_ID="${PSMA_FC70_FOLD}" GPU_ID="${PSMA_FC70_GPU}" PARENT_STAMP="${PARENT}" \
  TASK1_DPDNET_SKIP_PREPARE=1 TASK1_DPDNET_FDG_BEST="${DPD_FDG_BEST}" \
  TASK1_DPDNET_NUM_EPOCHS="${PSMA_FC70_EP}" TASK1_DPDNET_TRAIN_ITERS="${TR}" \
  TASK1_DPDNET_VAL_ITERS="${VAL}" TASK1_DPDNET_VAL_EVERY="${PSMA_FC70_VAL_EVERY}" \
  TASK1_BEST_BY=val_loss TASK1_DPDNET_BATCH_SIZE="${PSMA_FC70_BS}" \
  bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_fewshot50_onefold_bg.sh"

STAMP="${PARENT}_f${PSMA_FC70_FOLD}"
STOP="${WORK}/01_train_vis/TASK1_TRAIN_STOP_${STAMP}.txt"
rm -f "${STOP}" || true
nohup python3 "${MON}" --method dpdnet --parent-stamp "${PARENT}" --fold "${PSMA_FC70_FOLD}" \
  --base-ep "${PSMA_FC70_BASE_EP}" --val-every "${PSMA_FC70_VAL_EVERY}" \
  >"${VIS}/nohup_decline_mon_dpdnet_fc70.log" 2>&1 &

_wait_docker_up() {
  local stamp="$1" i
  for i in $(seq 1 36); do
    docker ps --format '{{.Names}}' 2>/dev/null | grep -qF "${stamp}" && return 0
    sleep 5
  done
  echo "[warn] docker never appeared for ${stamp}" >&2
  return 1
}
_wait_train_done() {
  local stamp="$1" stop="$2"
  _wait_docker_up "${stamp}" || return 1
  while [[ ! -f "${stop}" ]]; do
    docker ps --format '{{.Names}}' 2>/dev/null | grep -qF "${stamp}" || return 0
    sleep 30
  done
  return 0
}

_wait_train_done "${STAMP}" "${STOP}" || true

if [[ ! -f "${STOP}" ]]; then
  FOLD_ID="${PSMA_FC70_FOLD}" GPU_ID="${PSMA_FC70_GPU}" PARENT_STAMP="${PARENT}" \
    TASK1_DPDNET_SKIP_PREPARE=1 TASK1_DPDNET_FDG_BEST="${DPD_FDG_BEST}" \
    TASK1_DPDNET_NUM_EPOCHS="${PSMA_FC70_MAX_EP}" TASK1_DPDNET_TRAIN_ITERS="${TR}" \
    TASK1_DPDNET_VAL_ITERS="${VAL}" TASK1_DPDNET_VAL_EVERY="${PSMA_FC70_VAL_EVERY}" \
    TASK1_BEST_BY=ema_fg_dice TASK1_DPDNET_BATCH_SIZE="${PSMA_FC70_BS}" \
    TASK1_CONTINUE_TRAINING=1 TASK1_CONTINUE_FROM_BEST=1 \
    bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_fewshot50_onefold_bg.sh"
  _wait_train_done "${STAMP}" "${STOP}" || true
fi

DS240="Dataset240_DpDNet_PSMA_2ch"
TRAINER="${TRAINER:-STUNetTrainer_small_prompt}"
TF240="${TRAINER}__nnUNetPlans__3d_fullres"
FOLD_DIR="${WORK}/nnUNet_results/${STAMP}/${DS240}/${TF240}/fold_${PSMA_FC70_FOLD}"
[[ -d "${FOLD_DIR}" ]] || FOLD_DIR="${WORK}/nnUNet_results/${STAMP}/${DS240}/${TF240}/fold_0"
# dualenc may land under *_pretrain even if TRAINER was unset at check time
if [[ ! -d "${FOLD_DIR}" ]]; then
  for _tf in STUNetTrainer_small_prompt_pretrain__nnUNetPlans__3d_fullres \
             STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres; do
    _cand="${WORK}/nnUNet_results/${STAMP}/${DS240}/${_tf}/fold_${PSMA_FC70_FOLD}"
    [[ -d "${_cand}" ]] || _cand="${WORK}/nnUNet_results/${STAMP}/${DS240}/${_tf}/fold_0"
    if [[ -d "${_cand}" ]]; then
      FOLD_DIR="${_cand}"
      TF240="${_tf}"
      TRAINER="${_tf%%__*}"
      break
    fi
  done
fi
ckpt=""
for c in checkpoint_best.pth checkpoint_final.pth checkpoint_latest.pth; do
  [[ -f "${FOLD_DIR}/${c}" ]] && { ckpt="${c}"; break; }
done
if [[ -z "${ckpt}" ]]; then
  echo "[error] no Dataset240 ckpt after train (${FOLD_DIR}) — skip TEST20" >&2
  python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" --no-plot \
    --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"${PSMA_FC70_STAGE}\":{\"status\":\"pending\",\"note\":\"fc70 TEST20 missing ckpt\"}}},\"updated_note\":\"${BOARD_METHOD} fc70 skipped TEST20 (no ckpt)\"}" || true
  exit 1
fi

PARENT_STAMP="${PARENT}" TASK1_PSMA_FC70_FOLD="${PSMA_FC70_FOLD}" TASK1_PSMA_FC70_GPU="${PSMA_FC70_GPU}" \
  TRAINER="${TRAINER}" TASK1_BOARD_METHOD="${BOARD_METHOD}" \
  bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_test20_fc70_bg.sh"

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" --no-plot || true
echo "[dpdnet-fc70] ALL DONE ${PARENT}"
