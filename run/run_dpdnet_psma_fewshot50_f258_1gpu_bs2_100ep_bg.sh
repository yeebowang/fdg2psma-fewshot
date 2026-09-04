#!/usr/bin/env bash
# DpDNet PSMA fewshot50 · folds 2/5/8 · 1 GPU/fold · tr=25 · val=25 every 20ep · 100ep
# best=val_loss → checkpoint_best；训完后 TEST20 用各折最低 val_loss ckpt。
#
#   # after FDG done:
#   export TASK1_DPDNET_FDG_BEST=/path/to/checkpoint_final.pth   # prefer final>latest
#   bash ICLR2026/run/run_dpdnet_psma_fewshot50_f258_1gpu_bs2_100ep_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
ICLR_VIS="${CTRL}/ICLR2026/vis"
IMAGE="${TASK1_NNUNET_IMAGE:-autopet_baseline:latest}"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
mkdir -p "${ICLR_VIS}"

DATASET_ID="${DATASET_ID:-240}"
DS="Dataset${DATASET_ID}_DpDNet_PSMA_2ch"
TRAINER="${TRAINER:-STUNetTrainer_small_prompt}"
CONFIG="${CONFIG:-3d_fullres}"
TF="${TRAINER}__nnUNetPlans__${CONFIG}"
TOTAL_EPOCHS="${TASK1_DPDNET_NUM_EPOCHS:-${TASK1_NUM_EPOCHS:-100}}"
TRAIN_ITERS="${TASK1_DPDNET_TRAIN_ITERS:-${TASK1_TRAIN_ITERS_PER_EPOCH:-25}}"
VAL_ITERS="${TASK1_DPDNET_VAL_ITERS:-${TASK1_FS50_VAL_ITERS:-${TASK1_VAL_ITERS_PER_EPOCH:-25}}}"
VAL_EVERY="${TASK1_DPDNET_VAL_EVERY:-${TASK1_FS50_VAL_EVERY_N_EPOCHS:-${TASK1_VAL_EVERY_N_EPOCHS:-20}}}"
BATCH="${TASK1_DPDNET_BATCH_SIZE:-${TASK1_FIXED_BATCH_3D_FULLRES:-2}}"
FOLDS_CSV="${TASK1_FOLDS:-2,5,8}"
FOLD_GPUS_CSV="${TASK1_FOLD_GPUS:-2:0,5:1,8:3}"
BEST_BY="${TASK1_BEST_BY:-val_loss}"

FEWSHOT_N="${TASK1_FEWSHOT_N:-50}"
BOARD_STAGE="${TASK1_PSMA_BOARD_STAGE:-psma_fs${FEWSHOT_N}_f258}"
BOARD_METHOD="${TASK1_BOARD_METHOD:-dpdnet}"
SPLIT_DIR="${TASK1_FEWSHOT_SPLIT_DIR:-${ROOT}/ICLR2026/data/splits_mae_psma_fewshot${FEWSHOT_N}_9fold}"

# 1) prepare Dataset240 once
if [[ "${TASK1_DPDNET_SKIP_PREPARE:-0}" != "1" ]]; then
  docker run --rm --user root \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    --entrypoint python3 "${IMAGE}" \
    "${CTRL}/ICLR2026/scripts/prepare_dpdnet_psma_dataset240.py" \
      --work "${WORK}" \
      --dst-id "${DATASET_ID}" \
      --batch-size "${BATCH}" \
      --split-dir "${SPLIT_DIR}"
  docker run --rm --user root -v "${DATA}:${DATA}" --entrypoint bash "${IMAGE}" -lc \
    "chown -R $(id -u):$(id -g) '${WORK}/nnUNet_preprocessed/${DS}' '${WORK}/nnUNet_raw/${DS}' 2>/dev/null || true"
fi
[[ -f "${WORK}/nnUNet_preprocessed/${DS}/splits_final.json" ]] || {
  echo "[error] missing prepared ${DS}" >&2
  exit 1
}

_req="${TASK1_NNUNET_RESULTS_STAMP_NAME:-}"
if [[ -n "${_req}" && "${_req}" == *dpdnet*_psma_fs${FEWSHOT_N}* && "${_req}" != *_f[0-9] ]]; then
  PARENT="${_req}"
elif [[ "${BOARD_METHOD}" == "dpdnet_dualenc" ]]; then
  PARENT="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_dpdnet_dualenc_psma_fs${FEWSHOT_N}_f258_1gpu_bs${BATCH}_tr${TRAIN_ITERS}_val${VAL_ITERS}e${VAL_EVERY}_${TOTAL_EPOCHS}ep_gpu013"
else
  PARENT="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_dpdnet_psma_fs${FEWSHOT_N}_f258_1gpu_bs${BATCH}_tr${TRAIN_ITERS}_val${VAL_ITERS}e${VAL_EVERY}_${TOTAL_EPOCHS}ep_gpu013"
fi

IFS=',' read -r -a FOLDS <<< "${FOLDS_CSV}"
declare -A GPU_OF
IFS=',' read -r -a _pairs <<< "${FOLD_GPUS_CSV}"
for p in "${_pairs[@]}"; do
  GPU_OF["${p%%:*}"]="${p##*:}"
done

AGG_ROOT="${WORK}/nnUNet_results/${PARENT}"
PIPE_LOG="${ICLR_VIS}/nohup_dpdnet_psma_fs${FEWSHOT_N}_f258_${PARENT}.log"
META="${ICLR_VIS}/iclr2026_dpdnet_psma_fs${FEWSHOT_N}_f258_${PARENT}.txt"
mkdir -p "${AGG_ROOT}"
exec > >(tee -a "${PIPE_LOG}") 2>&1

{
  echo "job=iclr2026_dpdnet_psma_fewshot50_f258_1gpu"
  echo "PARENT=${PARENT}"
  echo "folds=${FOLDS_CSV} gpus=${FOLD_GPUS_CSV}"
  echo "epochs=${TOTAL_EPOCHS} tr=${TRAIN_ITERS} val=${VAL_ITERS} every=${VAL_EVERY} bs=${BATCH} best_by=${BEST_BY}"
} | tee "${META}"
echo "${PARENT}" > "${ICLR_VIS}/dpdnet_psma_fs${FEWSHOT_N}_f258_LAST_STAMP.txt"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"${BOARD_STAGE}\":{\"status\":\"running\",\"stamp\":\"${PARENT}\",\"bs\":${BATCH},\"bs_note\":\"per-GPU 1fold/GPU\",\"total_epochs\":${TOTAL_EPOCHS},\"train_iters\":${TRAIN_ITERS},\"val_iters\":${VAL_ITERS},\"online_val\":\"VAL${VAL_ITERS} every${VAL_EVERY}\",\"metric\":\"TEST20 Dice; best=val_loss\",\"note\":\"tr${TRAIN_ITERS}/val${VAL_ITERS}e${VAL_EVERY} · ${TOTAL_EPOCHS}ep · fs${FEWSHOT_N} · best=val_loss\"}}},\"updated_note\":\"${BOARD_METHOD} PSMA fs${FEWSHOT_N} f258 bs=${BATCH} tr${TRAIN_ITERS}/val${VAL_ITERS}e${VAL_EVERY} ${TOTAL_EPOCHS}ep\"}" || true

for fold in "${FOLDS[@]}"; do
  gpu="${GPU_OF[${fold}]:-}"
  [[ -n "${gpu}" ]] || { echo "[error] no GPU for fold ${fold}" >&2; exit 1; }
  echo "[dpdnet-psma] launch fold${fold} → GPU${gpu}"
  env -u TASK1_NNUNET_RESULTS_STAMP_NAME \
    FOLD_ID="${fold}" GPU_ID="${gpu}" PARENT_STAMP="${PARENT}" \
    TASK1_BASE="${DATA}" \
    TASK1_DPDNET_SKIP_PREPARE=1 \
    TASK1_DPDNET_NUM_EPOCHS="${TOTAL_EPOCHS}" \
    TASK1_DPDNET_TRAIN_ITERS="${TRAIN_ITERS}" \
    TASK1_DPDNET_VAL_ITERS="${VAL_ITERS}" \
    TASK1_DPDNET_VAL_EVERY="${VAL_EVERY}" \
    TASK1_VAL_EVERY_N_EPOCHS="${VAL_EVERY}" \
    TASK1_BEST_BY="${BEST_BY}" \
    TASK1_DPDNET_BATCH_SIZE="${BATCH}" \
    TASK1_DPDNET_FDG_BEST="${TASK1_DPDNET_FDG_BEST:-}" \
    TASK1_DPDNET_FDG_STAMP="${TASK1_DPDNET_FDG_STAMP:-}" \
    TASK1_DPDNET_FDG_FORCE_STAMP="${TASK1_DPDNET_FDG_FORCE_STAMP:-0}" \
    TASK1_DPDNET_FDG_TF="${TASK1_DPDNET_FDG_TF:-${TF}}" \
    TRAINER="${TRAINER}" \
    TASK1_DPDNET_SKIP_ENCODER_INIT="${TASK1_DPDNET_SKIP_ENCODER_INIT:-0}" \
    TASK1_DPDNET_CT_ENCODER="${TASK1_DPDNET_CT_ENCODER:-}" \
    TASK1_DPDNET_PET_ENCODER="${TASK1_DPDNET_PET_ENCODER:-}" \
    bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_fewshot50_onefold_bg.sh" &
  echo $! >"${AGG_ROOT}/launch_fold${fold}.pid"
  sleep 12
done

echo "[dpdnet-psma] waiting for fold trains (ep=${TOTAL_EPOCHS})…"

_fold_epoch() {
  local fold="$1"
  local stamp="${PARENT}_f${fold}"
  local fold_dir="${WORK}/nnUNet_results/${stamp}/${DS}/${TF}/fold_${fold}"
  python3 - <<PY
from pathlib import Path
import re
fd = Path("${fold_dir}")
if (fd / "checkpoint_final.pth").is_file():
    print(${TOTAL_EPOCHS})
    raise SystemExit
logs = sorted(fd.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
ep = 0
if logs:
    for line in logs[0].read_text(errors="ignore").splitlines():
        m = re.search(r"Epoch[: ]+(\d+)", line, re.I)
        if m:
            ep = max(ep, int(m.group(1)))
print(ep)
PY
}

while true; do
  ok=0
  for fold in "${FOLDS[@]}"; do
    ep="$(_fold_epoch "${fold}")"
    echo "[dpdnet-psma] fold${fold} ep=${ep}/${TOTAL_EPOCHS}"
    [[ "${ep}" -ge "${TOTAL_EPOCHS}" ]] && ok=$((ok + 1))
  done
  [[ "${ok}" -ge "${#FOLDS[@]}" ]] && break
  sleep 90
done

if [[ "${TASK1_SKIP_TEST20_AT_END:-0}" != "1" ]]; then
echo "[dpdnet-psma] all folds trained → TEST20 (checkpoint_best = min val_loss)"
for fold in "${FOLDS[@]}"; do
  TASK1_CRASH_MONITOR_STAGE="dpdnet_psma_f${fold}_before_eval" \
  TASK1_NNUNET_RESULTS_STAMP_NAME="${PARENT}_f${fold}" \
    bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
done

export PARENT_STAMP="${PARENT}"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${PARENT}"
export TASK1_FOLDS="${FOLDS_CSV}"
export TASK1_FOLD_GPUS="${FOLD_GPUS_CSV}"
export TASK1_UDA_PRED_PER_GPU="${TASK1_UDA_PRED_PER_GPU:-5}"
export TASK1_TEST_SKIP_DONE=1
bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_test20_f258_parallel.sh"

echo "[dpdnet-psma] ALL DONE → ${AGG_ROOT}/aggregate_test20_dice_f258.json"
fi
