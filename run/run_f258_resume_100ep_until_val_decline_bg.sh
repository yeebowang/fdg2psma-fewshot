#!/usr/bin/env bash
# Resume F258 PSMA folds whose best ckpt is @100ep; stop each fold on first val-Dice decline.
# Only folds with best@100 are trained (e.g. nnUNet f5 only; DpDNet/SegAnyPET all 3).
#
#   bash ICLR2026/run/run_f258_resume_100ep_until_val_decline_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
MON="${CTRL}/ICLR2026/scripts/monitor_val_dice_decline_stop.py"

BASE_EP="${TASK1_RESUME_BASE_EP:-100}"
MAX_EP="${TASK1_RESUME_MAX_EPOCHS:-300}"
VAL_EVERY="${TASK1_FS50_VAL_EVERY_N_EPOCHS:-20}"
FOLD_GPUS="${TASK1_FOLD_GPUS:-2:0,5:1,8:3}"

NN_PARENT="${NN_PARENT:-20260818_010404_iclr2026_nnunet_psma_fs50_f258_1gpu_bs2_tr25_val25e20_100ep_gpu013}"
DPD_PARENT="${DPD_PARENT:-20260817_210749_iclr2026_dpdnet_psma_fs50_f258_1gpu_bs2_tr25_val25e20_100ep_gpu013}"
SEG_STAMP="${SEG_STAMP:-20260817_114450_iclr2026_seganypet_fs50_from_fdg_f258_gpu013}"

# folds to resume (detected: nn f5; dpd 2,5,8; seg 2,5,8)
NN_FOLDS="${NN_RESUME_FOLDS:-5}"
DPD_FOLDS="${DPD_RESUME_FOLDS:-2,5,8}"
SEG_FOLDS="${SEG_RESUME_FOLDS:-2,5,8}"

PIPE_LOG="${VIS}/nohup_f258_resume_100ep_until_val_decline.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

declare -A GPU_OF
IFS=',' read -r -a _pairs <<< "${FOLD_GPUS}"
for p in "${_pairs[@]}"; do
  GPU_OF["${p%%:*}"]="${p##*:}"
done

_start_monitor() {
  local method="$1" parent="$2" fold="$3"
  local log="${VIS}/nohup_decline_mon_${method}_f${fold}.log"
  nohup python3 "${MON}" \
    --method "${method}" \
    --parent-stamp "${parent}" \
    --fold "${fold}" \
    --base-ep "${BASE_EP}" \
    --val-every "${VAL_EVERY}" \
    >"${log}" 2>&1 &
  echo $! > "${VIS}/decline_mon_${method}_f${fold}.pid"
  echo "[resume] monitor ${method} f${fold} pid=$(cat "${VIS}/decline_mon_${method}_f${fold}.pid") log=${log}"
}

_wait_fold_stamps() {
  local label="$1"
  shift
  local stamps=("$@")
  echo "[resume] waiting ${label}: ${stamps[*]}"
  while true; do
    local done=0
    for st in "${stamps[@]}"; do
      if [[ -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${st}.txt" ]]; then
        done=$((done + 1))
        continue
      fi
      if ! pgrep -af "${st}" >/dev/null 2>&1 \
         && ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qF "${st}"; then
        done=$((done + 1))
      fi
    done
    echo "[resume] ${label} finished=${done}/${#stamps[@]}"
    [[ "${done}" -ge "${#stamps[@]}" ]] && break
    sleep 60
  done
}

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" \
  --patch-json '{"updated_note":"F258 resume@100ep until val Dice decline (100/100 folds only)"}' \
  --no-plot || true

echo "[resume] === nnUNet PARENT=${NN_PARENT} folds=${NN_FOLDS} (skip f2/f8 best@80) ==="
IFS=',' read -r -a _nnf <<< "${NN_FOLDS}"
nn_stamps=()
for fold in "${_nnf[@]}"; do
  gpu="${GPU_OF[${fold}]:-}"
  [[ -n "${gpu}" ]] || { echo "[error] no GPU for nn fold ${fold}" >&2; exit 1; }
  st="${NN_PARENT}_f${fold}"
  nn_stamps+=("${st}")
  rm -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${st}.txt"
  FOLD_ID="${fold}" GPU_ID="${gpu}" PARENT_STAMP="${NN_PARENT}" \
    TASK1_NUM_EPOCHS="${MAX_EP}" \
    TASK1_LR_SCHEDULE_NUM_EPOCHS="${BASE_EP}" \
    TASK1_TRAIN_ITERS_PER_EPOCH=25 \
    TASK1_VAL_ITERS_PER_EPOCH=25 \
    TASK1_FS50_VAL_ITERS=25 \
    TASK1_VAL_EVERY_N_EPOCHS="${VAL_EVERY}" \
    TASK1_FS50_VAL_EVERY_N_EPOCHS="${VAL_EVERY}" \
    TASK1_FIXED_BATCH_3D_FULLRES=2 \
    TASK1_BEST_BY=ema_fg_dice \
    TASK1_VAL_LOSS_ONLY=0 \
    TASK1_CONTINUE_TRAINING=1 \
    TASK1_CONTINUE_FROM_BEST=1 \
    TASK1_CONTINUE_FROM_LATEST=0 \
    TASK1_CONTINUE_PICK_NEWER=1 \
    bash "${CTRL}/ICLR2026/run/run_nnunet_psma_fewshot50_onefold_bg.sh"
  sleep 8
  _start_monitor nnunet "${NN_PARENT}" "${fold}"
done
_wait_fold_stamps nnunet "${nn_stamps[@]}"
for st in "${nn_stamps[@]}"; do
  TASK1_NNUNET_RESULTS_STAMP_NAME="${st}" bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
done

echo "[resume] === DpDNet PARENT=${DPD_PARENT} folds=${DPD_FOLDS} ==="
IFS=',' read -r -a _dpf <<< "${DPD_FOLDS}"
dpd_stamps=()
for fold in "${_dpf[@]}"; do
  gpu="${GPU_OF[${fold}]:-}"
  [[ -n "${gpu}" ]] || { echo "[error] no GPU for dpd fold ${fold}" >&2; exit 1; }
  st="${DPD_PARENT}_f${fold}"
  dpd_stamps+=("${st}")
  rm -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${st}.txt"
  FOLD_ID="${fold}" GPU_ID="${gpu}" PARENT_STAMP="${DPD_PARENT}" \
    TASK1_DPDNET_NUM_EPOCHS="${MAX_EP}" \
    TASK1_DPDNET_TRAIN_ITERS=25 \
    TASK1_DPDNET_VAL_ITERS=25 \
    TASK1_DPDNET_VAL_EVERY="${VAL_EVERY}" \
    TASK1_BEST_BY=ema_fg_dice \
    TASK1_DPDNET_SKIP_PREPARE=1 \
    bash "${CTRL}/ICLR2026/run/run_dpdnet_psma_fewshot50_onefold_bg.sh"
  sleep 8
  _start_monitor dpdnet "${DPD_PARENT}" "${fold}"
done
_wait_fold_stamps dpdnet "${dpd_stamps[@]}"
for st in "${dpd_stamps[@]}"; do
  TASK1_NNUNET_RESULTS_STAMP_NAME="${st}" bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
done

echo "[resume] === SegAnyPET STAMP=${SEG_STAMP} folds=${SEG_FOLDS} ==="
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
SEG_CODE="${CTRL}/ICLR2026/third_party/SegAnyPET/code"
SEG_PIP="${CTRL}/ICLR2026/third_party/seganypet_pip"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"
DATA_ROOT="${DATA}/task1_train_workspace/seganypet_fewshot50_f258"
OUT_ROOT="${REPO}/runs/${SEG_STAMP}"
SEG_CKPT="${TASK1_SEGANY_CKPT:-${REPO}/weights/seganypet/seganypet_lesion.pth}"
IFS=',' read -r -a _sgf <<< "${SEG_FOLDS}"
seg_cnames=()
for fold in "${_sgf[@]}"; do
  gpu="${GPU_OF[${fold}]:-}"
  [[ -n "${gpu}" ]] || { echo "[error] no GPU for seg fold ${fold}" >&2; exit 1; }
  rm -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${SEG_STAMP}.txt"
  fold_data="${DATA_ROOT}/fold${fold}"
  out_dir="${OUT_ROOT}/seganypet/fold${fold}"
  log="${VIS}/nohup_seganypet_resume_f${fold}_${SEG_STAMP}.log"
  cname="seganypet_fs50_f${fold}_${SEG_STAMP}"
  seg_cnames+=("${cname}")
  docker rm -f "${cname}" >/dev/null 2>&1 || true
  nohup docker run --rm \
    --name "${cname}" \
    --gpus "device=${gpu}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONPATH="${SEG_PIP}:${SEG_CODE}:${CTRL}/ICLR2026/scripts" \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    -w "${SEG_CODE}" --shm-size=8g "${IMAGE}" \
    python3 "${CTRL}/ICLR2026/scripts/seganypet_fewshot_finetune.py" \
      --data-root "${fold_data}" \
      --checkpoint "${SEG_CKPT}" \
      --out-dir "${out_dir}" \
      --epochs "${MAX_EP}" \
      --batch-size 2 --accumulation-steps 20 --num-workers 6 \
      --val-interval "${VAL_EVERY}" --val-clicks 5 --val-max-cases 15 \
      --lr-mode official --lr 8e-4 --milestones 60,85 --click-max 11 --no-dataparallel \
    >"${log}" 2>&1 &
  sleep 8
  _start_monitor seganypet "${SEG_STAMP}" "${fold}"
done
while true; do
  done=0
  for cname in "${seg_cnames[@]}"; do
    if [[ -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${SEG_STAMP}.txt" ]]; then
      done=$((done + 1)); continue
    fi
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qxF "${cname}"; then
      done=$((done + 1))
    fi
  done
  echo "[resume] seganypet finished=${done}/${#seg_cnames[@]}"
  [[ "${done}" -ge "${#seg_cnames[@]}" ]] && break
  sleep 60
done

echo "[resume] === TEST20 refresh (post resume) ==="
# nnUNet: re-test resumed folds only; aggregate always needs all 3 score_detail.json
NN_EVAL="${WORK}/nnUNet_results/${NN_PARENT}/psma_test20_eval"
IFS=',' read -r -a _nnf <<< "${NN_FOLDS}"
for fold in "${_nnf[@]}"; do
  rm -f "${NN_EVAL}/fold${fold}/score_detail.json"
done
TASK1_FOLDS="2,5,8" TASK1_TEST_SKIP_DONE=1 \
  PARENT_STAMP="${NN_PARENT}" \
  bash "${CTRL}/ICLR2026/run/run_nnunet_psma_test20_f258_parallel.sh" || true

TASK1_FOLDS="2,5,8" TASK1_TEST_SKIP_DONE=0 PARENT_STAMP="${DPD_PARENT}" \
  bash "${CTRL}/ICLR2026/run/run_dpdnet_psma_test20_f258_parallel.sh" || true

TASK1_FOLDS="2,5,8" STAMP="${SEG_STAMP}" \
  bash "${CTRL}/ICLR2026/run/run_eval_seganypet_psma_test20_f258_bg.sh" || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" \
  --patch-json '{"updated_note":"F258 resume-until-decline DONE; TEST20 [epN] + mean refreshed"}' || true

echo "[resume] ALL DONE log=${PIPE_LOG}"
