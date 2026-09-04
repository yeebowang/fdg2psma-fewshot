#!/usr/bin/env bash
# PSMA fs50 / fs10 / fs5: add folds 0,1,3,4,6,7 on existing stamps → full 9-fold (fs50 max=9).
# Reuses board stamps; trains missing folds (100ep) + decline resume + TEST20 on all 9 folds.
#
#   bash ICLR2026/run/run_aligned_psma_extra_folds_9fold_pipeline_bg.sh
#   TASK1_FEWSHOT_LIST=50,10,5 TASK1_METHODS=mae,monai,dpdnet,seganypet,nnunet ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${ROOT}/ICLR2026/vis"
REPO="${ROOT}/ICLR2026/3D-MAE-PET-CT"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"

EXTRA_FOLDS="${TASK1_EXTRA_FOLDS:-0,1,3,4,6,7}"
ALL_FOLDS="${TASK1_ALL_FOLDS:-0,1,2,3,4,5,6,7,8}"
EXTRA_FOLD_GPUS="${TASK1_EXTRA_FOLD_GPUS:-0:0,1:1,3:3,4:0,6:1,7:3}"
ALL_FOLD_GPUS="${TASK1_ALL_FOLD_GPUS:-0:0,1:1,2:0,3:3,4:0,5:1,6:3,7:0,8:3}"
FEWSHOT_LIST="${TASK1_FEWSHOT_LIST:-50,10,5}"
METHODS="${TASK1_METHODS:-mae,monai,dpdnet,seganypet,nnunet}"

FDG_STAMP="${TASK1_UDA_FDG_STAMP:-20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep}"
FDG_BEST="${TASK1_UDA_FDG_BEST:-/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth}"
MAE_FDG_SEG="${TASK1_MAE_FDG_SEG_CKPT:-${REPO}/runs/20260812_072719_iclr2026_mae_fdg_swinbase_gpu013_bs6_tr70_val10_100ep/best_seg_fdg_mae.pth}"
MONAI_FDG_SEG="${TASK1_MONAI_FDG_SEG_CKPT:-${REPO}/runs/20260816_214921_iclr2026_monai_fdg_swinvit_1gpu_bs6_tr70_val10_100ep/best_seg_fdg_monai.pth}"
DPD_FDG_STAMP="${TASK1_DPDNET_FDG_STAMP:-20260817_165250_iclr2026_dpdnet_fdg_2gpu_bs3_gbs6_n6_tr70_val0_169ep_gpu01}"
DPD_FDG_BEST="${TASK1_DPDNET_FDG_BEST:-/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/20260817_165250_iclr2026_dpdnet_fdg_2gpu_bs3_gbs6_n6_tr70_val0_169ep_gpu01/Dataset239_DpDNet_FDG_2ch/STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth}"
SEG_FDG_BEST="${TASK1_SEGANY_CKPT:-${REPO}/runs/20260817_041526_iclr2026_seganypet_fdg_3gpu_bs6_gpu013/seganypet_fdg/best.pth}"

PSMA_EP="${TASK1_NUM_EPOCHS:-100}"
PSMA_TR="${TASK1_TRAIN_ITERS_PER_EPOCH:-25}"
PSMA_VAL="${TASK1_FS50_VAL_ITERS:-25}"
PSMA_EVERY="${TASK1_FS50_VAL_EVERY_N_EPOCHS:-20}"
PSMA_BS="${TASK1_FIXED_BATCH_3D_FULLRES:-2}"

PIPE_LOG="${VIS}/nohup_aligned_psma_extra_folds_9fold_pipeline.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[extra-9fold] fs=${FEWSHOT_LIST} methods=${METHODS} extra=${EXTRA_FOLDS} → all=${ALL_FOLDS}"

_split_dir() {
  echo "${ROOT}/ICLR2026/data/splits_mae_psma_fewshot${1}_9fold"
}

_ensure_splits() {
  local n="$1" dir
  dir="$(_split_dir "${n}")"
  [[ -f "${dir}/fold0_nnunet.json" ]] || python3 "${ROOT}/ICLR2026/scripts/export_mae_psma_fewshot50_9fold.py" \
    --n-shot "${n}" --out-dir "${dir}" --seed 42
}

_board_stamp() {
  local method="$1" stage="$2"
  python3 - <<PY
import json
from pathlib import Path
b = json.loads(Path("${BOARD}").read_text(encoding="utf-8"))
print((b.get("methods", {}).get("${method}", {}).get("${stage}") or {}).get("stamp", ""))
PY
}

_last_stamp() {
  local f="${VIS}/${1}"
  [[ -f "${f}" ]] && tr -d '[:space:]' < "${f}" || true
}

_resolve_stamp() {
  local method_key="$1" stage="$2" last_file="$3"
  local s
  s="$(_board_stamp "${method_key}" "${stage}")"
  [[ -n "${s}" ]] || s="$(_last_stamp "${last_file}")"
  echo "${s}"
}

_board_running() {
  local method_key="$1" stage="$2" note="$3"
  python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD}" --no-plot \
    --patch-json "{\"methods\":{\"${method_key}\":{\"${stage}\":{\"status\":\"running\",\"note\":\"${note}\"}}},\"updated_note\":\"${note}\"}" || true
}

_run_mae_extra() {
  local n="$1"
  local stage="psma_fs${n}_f258"
  local split_dir stamp
  split_dir="$(_split_dir "${n}")"
  _ensure_splits "${n}"
  stamp="$(_resolve_stamp "mae_swinunetr" "${stage}" "mae_psma_fs${n}_f258_LAST_STAMP.txt")"
  [[ -n "${stamp}" ]] || { echo "[error] MAE stamp missing fs${n}" >&2; return 1; }

  export TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}"
  export TASK1_FEWSHOT_SPLIT_DIR="${split_dir}" TASK1_MAE_FDG_SEG_CKPT="${MAE_FDG_SEG}"
  export TASK1_MAE_NUM_EPOCHS="${PSMA_EP}" TASK1_MAE_BATCH_SIZE="${PSMA_BS}"
  export TASK1_MAE_FEWSHOT_FOLDS_CSV="${EXTRA_FOLDS}"
  export TASK1_NNUNET_RESULTS_STAMP_NAME="${stamp}"

  echo "[extra-9fold] MAE fs${n} stamp=${stamp} train folds=${EXTRA_FOLDS}"
  _board_running "mae_swinunetr" "${stage}" "9fold extra · MAE fs${n} · folds ${EXTRA_FOLDS}"
  bash "${ROOT}/ICLR2026/run/run_mae_psma_fewshot50_f258_from_fdg_seg_bg.sh"

  METHOD=mae STAMP="${stamp}" TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}" \
    TASK1_MAE_FEWSHOT_FOLDS_CSV="${ALL_FOLDS}" TASK1_FOLD_GPUS="${ALL_FOLD_GPUS}" TASK1_TEST_SKIP_DONE=0 \
    bash "${ROOT}/ICLR2026/run/run_eval_psma_test20_f258_bg.sh"
}

_run_monai_extra() {
  local n="$1"
  local stage="psma_fs${n}_f258"
  local split_dir stamp
  split_dir="$(_split_dir "${n}")"
  _ensure_splits "${n}"
  stamp="$(_resolve_stamp "monai_swinvit" "${stage}" "monai_psma_fs${n}_f258_LAST_STAMP.txt")"
  [[ -n "${stamp}" ]] || { echo "[error] MONAI stamp missing fs${n}" >&2; return 1; }

  export TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}"
  export TASK1_FEWSHOT_SPLIT_DIR="${split_dir}" TASK1_MONAI_FDG_SEG_CKPT="${MONAI_FDG_SEG}"
  export TASK1_MAE_NUM_EPOCHS="${PSMA_EP}" TASK1_MAE_BATCH_SIZE="${PSMA_BS}"
  export TASK1_MAE_FEWSHOT_FOLDS_CSV="${EXTRA_FOLDS}"
  export TASK1_NNUNET_RESULTS_STAMP_NAME="${stamp}"

  echo "[extra-9fold] MONAI fs${n} stamp=${stamp} train folds=${EXTRA_FOLDS}"
  _board_running "monai_swinvit" "${stage}" "9fold extra · MONAI fs${n} · folds ${EXTRA_FOLDS}"
  bash "${ROOT}/ICLR2026/run/run_monai_psma_fewshot50_f258_from_fdg_seg_bg.sh"

  METHOD=monai STAMP="${stamp}" TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}" \
    TASK1_MAE_FEWSHOT_FOLDS_CSV="${ALL_FOLDS}" TASK1_FOLD_GPUS="${ALL_FOLD_GPUS}" TASK1_TEST_SKIP_DONE=0 \
    bash "${ROOT}/ICLR2026/run/run_eval_psma_test20_f258_bg.sh"
}

_run_nnunet_extra() {
  local n="$1"
  local stage="psma_fs${n}_f258"
  local split_dir parent
  split_dir="$(_split_dir "${n}")"
  _ensure_splits "${n}"
  parent="$(_resolve_stamp "nnunet" "${stage}" "nnunet_psma_fs${n}_f258_LAST_STAMP.txt")"
  [[ -n "${parent}" ]] || { echo "[error] nnUNet PARENT missing fs${n}" >&2; return 1; }

  export TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}"
  export TASK1_FEWSHOT_SPLIT_DIR="${split_dir}"
  export TASK1_UDA_FDG_STAMP="${FDG_STAMP}" TASK1_UDA_FDG_BEST="${FDG_BEST}"
  export TASK1_NUM_EPOCHS="${PSMA_EP}" TASK1_TRAIN_ITERS_PER_EPOCH="${PSMA_TR}"
  export TASK1_FS50_VAL_ITERS="${PSMA_VAL}" TASK1_FS50_VAL_EVERY_N_EPOCHS="${PSMA_EVERY}"
  export TASK1_VAL_EVERY_N_EPOCHS="${PSMA_EVERY}" TASK1_VAL_ITERS_PER_EPOCH="${PSMA_VAL}"
  export TASK1_FIXED_BATCH_3D_FULLRES="${PSMA_BS}" TASK1_BEST_BY=val_loss TASK1_VAL_LOSS_ONLY=1
  export TASK1_FOLDS="${EXTRA_FOLDS}" TASK1_FOLD_GPUS="${EXTRA_FOLD_GPUS}" TASK1_SKIP_TEST20_AT_END=1
  export TASK1_NNUNET_RESULTS_STAMP_NAME="${parent}"

  echo "[extra-9fold] nnUNet fs${n} PARENT=${parent} train folds=${EXTRA_FOLDS}"
  _board_running "nnunet" "${stage}" "9fold extra · nnUNet fs${n} · folds ${EXTRA_FOLDS}"
  bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh"

  NN_PARENT="${parent}" TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}" \
    TASK1_FEWSHOT_SPLIT_DIR="${split_dir}" TASK1_UDA_FDG_STAMP="${FDG_STAMP}" TASK1_UDA_FDG_BEST="${FDG_BEST}" \
    TASK1_FOLD_GPUS="${EXTRA_FOLD_GPUS}" NN_RESUME_FOLDS="${EXTRA_FOLDS}" \
    bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fewshot_f258_decline_and_test_bg.sh" || true

  export PARENT_STAMP="${parent}" TASK1_NNUNET_RESULTS_STAMP_NAME="${parent}"
  export TASK1_FOLDS="${ALL_FOLDS}" TASK1_FOLD_GPUS="${ALL_FOLD_GPUS}" TASK1_TEST_SKIP_DONE=0
  bash "${ROOT}/ICLR2026/run/run_nnunet_psma_test20_f258_parallel.sh"
}

_run_dpdnet_extra() {
  local n="$1"
  local stage="psma_fs${n}_f258"
  local split_dir parent
  split_dir="$(_split_dir "${n}")"
  _ensure_splits "${n}"
  parent="$(_resolve_stamp "dpdnet" "${stage}" "dpdnet_psma_fs${n}_f258_LAST_STAMP.txt")"
  [[ -n "${parent}" ]] || { echo "[error] DpDNet PARENT missing fs${n}" >&2; return 1; }

  export TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}"
  export TASK1_FEWSHOT_SPLIT_DIR="${split_dir}"
  export TASK1_DPDNET_FDG_STAMP="${DPD_FDG_STAMP}" TASK1_DPDNET_FDG_BEST="${DPD_FDG_BEST}"
  export TASK1_DPDNET_NUM_EPOCHS="${PSMA_EP}" TASK1_DPDNET_TRAIN_ITERS="${PSMA_TR}"
  export TASK1_DPDNET_VAL_ITERS="${PSMA_VAL}" TASK1_DPDNET_VAL_EVERY="${PSMA_EVERY}"
  export TASK1_DPDNET_BATCH_SIZE="${PSMA_BS}" TASK1_BEST_BY=val_loss
  export TASK1_FOLDS="${EXTRA_FOLDS}" TASK1_FOLD_GPUS="${EXTRA_FOLD_GPUS}" TASK1_SKIP_TEST20_AT_END=1
  export TASK1_DPDNET_SKIP_PREPARE=1 TASK1_NNUNET_RESULTS_STAMP_NAME="${parent}"

  echo "[extra-9fold] DpDNet fs${n} PARENT=${parent} train folds=${EXTRA_FOLDS}"
  _board_running "dpdnet" "${stage}" "9fold extra · DpDNet fs${n} · folds ${EXTRA_FOLDS}"
  bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_fewshot50_f258_1gpu_bs2_100ep_bg.sh"

  DPD_PARENT="${parent}" TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}" \
    TASK1_FOLD_GPUS="${EXTRA_FOLD_GPUS}" DPD_RESUME_FOLDS="${EXTRA_FOLDS}" \
    bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_fewshot_f258_decline_and_test_bg.sh" || true

  export PARENT_STAMP="${parent}" TASK1_FOLDS="${ALL_FOLDS}" TASK1_FOLD_GPUS="${ALL_FOLD_GPUS}" TASK1_TEST_SKIP_DONE=0
  bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_test20_f258_parallel.sh"
}

_run_seganypet_extra() {
  local n="$1"
  local stage="psma_fs${n}_f258"
  local split_dir data_root stamp
  split_dir="$(_split_dir "${n}")"
  data_root="${DATA}/task1_train_workspace/seganypet_fewshot${n}_f258"
  _ensure_splits "${n}"
  stamp="$(_resolve_stamp "seganypet" "${stage}" "seganypet_psma_fs${n}_f258_LAST_STAMP.txt")"
  [[ -n "${stamp}" ]] || { echo "[error] SegAnyPET stamp missing fs${n}" >&2; return 1; }

  export TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}"
  export TASK1_FEWSHOT_SPLIT_DIR="${split_dir}" TASK1_SEGANY_DATA_ROOT="${data_root}"
  export TASK1_SEGANY_CKPT="${SEG_FDG_BEST}" TASK1_SEGANY_EPOCHS="${PSMA_EP}" TASK1_SEGANY_BATCH_SIZE="${PSMA_BS}"
  export TASK1_SEGANY_FOLDS_CSV="${EXTRA_FOLDS}"
  export TASK1_NNUNET_RESULTS_STAMP_NAME="${stamp}"

  echo "[extra-9fold] SegAnyPET fs${n} stamp=${stamp} train folds=${EXTRA_FOLDS}"
  _board_running "seganypet" "${stage}" "9fold extra · SegAnyPET fs${n} · folds ${EXTRA_FOLDS}"
  bash "${ROOT}/ICLR2026/run/run_seganypet_fewshot50_f258_bg.sh"

  SEG_STAMP="${stamp}" TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}" \
    TASK1_SEGANY_DATA_ROOT="${data_root}" TASK1_SEGANY_CKPT="${SEG_FDG_BEST}" \
    TASK1_FOLD_GPUS="${EXTRA_FOLD_GPUS}" SEG_RESUME_FOLDS="${EXTRA_FOLDS}" \
    bash "${ROOT}/ICLR2026/run/run_seganypet_psma_fewshot_f258_decline_and_test_bg.sh" || true

  SEG_STAMP="${stamp}" TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}" \
    TASK1_SEGANY_FOLDS_CSV="${ALL_FOLDS}" TASK1_FOLD_GPUS="${ALL_FOLD_GPUS}" TASK1_TEST_SKIP_DONE=0 \
    bash "${ROOT}/ICLR2026/run/run_eval_seganypet_psma_test20_f258_bg.sh"
}

IFS=',' read -r -a _fs <<< "${FEWSHOT_LIST}"
for n in "${_fs[@]}"; do
  n="$(echo "${n}" | tr -d '[:space:]')"
  [[ -n "${n}" ]] || continue
  bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true

  if [[ "${METHODS}" == *mae* || "${METHODS}" == "all" ]]; then _run_mae_extra "${n}"; fi
  if [[ "${METHODS}" == *monai* || "${METHODS}" == "all" ]]; then _run_monai_extra "${n}"; fi
  if [[ "${METHODS}" == *dpdnet* || "${METHODS}" == "all" ]]; then _run_dpdnet_extra "${n}"; fi
  if [[ "${METHODS}" == *seganypet* || "${METHODS}" == "all" ]]; then _run_seganypet_extra "${n}"; fi
  if [[ "${METHODS}" == *nnunet* || "${METHODS}" == "all" ]]; then _run_nnunet_extra "${n}"; fi
done

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"updated_note\":\"PSMA fs50/fs10/fs5 extra folds → 9fold DONE · methods=${METHODS}\"}" || true

{
  echo "done_at=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "status=ok"
  echo "extra_folds=${EXTRA_FOLDS}"
  echo "all_folds=${ALL_FOLDS}"
  echo "fewshots=${FEWSHOT_LIST}"
  echo "methods=${METHODS}"
} > "${VIS}/TASK1_PSMA_EXTRA_FOLDS_9FOLD_DONE.txt"

echo "[extra-9fold] ALL DONE fs=${FEWSHOT_LIST} methods=${METHODS}"
