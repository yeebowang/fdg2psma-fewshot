#!/usr/bin/env bash
# SegAnyPET PSMA f258: resume from ep100 latest.pth until first val_dice decline.
#
#   bash ICLR2026/run/run_seganypet_resume_until_decline_bg.sh
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
SEG_STAMP="${SEG_STAMP:-20260817_114450_iclr2026_seganypet_fs50_from_fdg_f258_gpu013}"
SEG_FOLDS="${SEG_RESUME_FOLDS:-2,5,8}"

PIPE_LOG="${VIS}/nohup_seganypet_resume_until_decline.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

declare -A GPU_OF
IFS=',' read -r -a _pairs <<< "${FOLD_GPUS}"
for p in "${_pairs[@]}"; do
  GPU_OF["${p%%:*}"]="${p##*:}"
done

_start_monitor() {
  local fold="$1"
  local log="${VIS}/nohup_decline_mon_seganypet_f${fold}.log"
  nohup python3 "${MON}" \
    --method seganypet \
    --parent-stamp "${SEG_STAMP}" \
    --fold "${fold}" \
    --base-ep "${BASE_EP}" \
    --val-every "${VAL_EVERY}" \
    >"${log}" 2>&1 &
  echo $! > "${VIS}/decline_mon_seganypet_f${fold}.pid"
  echo "[seg-resume] monitor f${fold} pid=$(cat "${VIS}/decline_mon_seganypet_f${fold}.pid")"
}

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" \
  --patch-json '{"updated_note":"SegAnyPET resume@100ep until val Dice decline"}' \
  --no-plot || true

REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
SEG_CODE="${CTRL}/ICLR2026/third_party/SegAnyPET/code"
SEG_PIP="${CTRL}/ICLR2026/third_party/seganypet_pip"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"
DATA_ROOT="${DATA}/task1_train_workspace/seganypet_fewshot50_f258"
OUT_ROOT="${REPO}/runs/${SEG_STAMP}"
SEG_CKPT="${TASK1_SEGANY_CKPT:-${REPO}/runs/20260817_041526_iclr2026_seganypet_fdg_3gpu_bs6_gpu013/seganypet_fdg/best.pth}"

echo "[seg-resume] STAMP=${SEG_STAMP} folds=${SEG_FOLDS} base_ep=${BASE_EP} max_ep=${MAX_EP}"

IFS=',' read -r -a _sgf <<< "${SEG_FOLDS}"
seg_cnames=()
for fold in "${_sgf[@]}"; do
  gpu="${GPU_OF[${fold}]:-}"
  [[ -n "${gpu}" ]] || { echo "[error] no GPU for seg fold ${fold}" >&2; exit 1; }
  fold_data="${DATA_ROOT}/fold${fold}"
  out_dir="${OUT_ROOT}/seganypet/fold${fold}"
  latest="${out_dir}/latest.pth"
  [[ -f "${latest}" ]] || { echo "[error] missing ${latest} (need ep100 resume seed)" >&2; exit 1; }
  rm -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${SEG_STAMP}.txt"
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
  _start_monitor "${fold}"
done

while true; do
  done=0
  for cname in "${seg_cnames[@]}"; do
    if [[ -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${SEG_STAMP}.txt" ]]; then
      done=$((done + 1))
      continue
    fi
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qxF "${cname}"; then
      done=$((done + 1))
    fi
  done
  echo "[seg-resume] finished=${done}/${#seg_cnames[@]}"
  [[ "${done}" -ge "${#seg_cnames[@]}" ]] && break
  sleep 60
done

if [[ "${TASK1_SKIP_TEST20:-0}" == "1" ]]; then
  echo "[seg-resume] skip TEST20 (partial fold resume)"
else
  echo "[seg-resume] TEST20 refresh"
  TASK1_FOLDS="2,5,8" STAMP="${SEG_STAMP}" \
    bash "${CTRL}/ICLR2026/run/run_eval_seganypet_psma_test20_f258_bg.sh" || true

  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD}" \
    --patch-json '{"updated_note":"SegAnyPET resume-until-decline DONE; TEST20 refreshed"}' || true
fi

echo "[seg-resume] DONE log=${PIPE_LOG}"
