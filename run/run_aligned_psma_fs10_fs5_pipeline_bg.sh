#!/usr/bin/env bash
# PSMA fs10 + fs5 f258 pipeline (skip FDG — reuse board FDG ckpt).
# Methods: mae, monai, dpdnet, seganypet, nnunet (default skips nnunet if already done).
#
#   TASK1_METHODS=mae,monai,dpdnet,seganypet bash ICLR2026/run/run_aligned_psma_fs10_fs5_pipeline_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${ROOT}/ICLR2026/vis"
REPO="${ROOT}/ICLR2026/3D-MAE-PET-CT"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"

FEWSHOT_LIST="${TASK1_FEWSHOT_LIST:-10,5}"
METHODS="${TASK1_METHODS:-mae,monai,dpdnet,seganypet}"

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

PIPE_LOG="${VIS}/nohup_aligned_psma_fs10_fs5_pipeline.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[pipeline] fs10/fs5 methods=${METHODS} fewshots=${FEWSHOT_LIST}"

_split_dir() {
  echo "${ROOT}/ICLR2026/data/splits_mae_psma_fewshot${1}_9fold"
}

_ensure_splits() {
  local n="$1" dir
  dir="$(_split_dir "${n}")"
  [[ -f "${dir}/fold0_nnunet.json" ]] || python3 "${ROOT}/ICLR2026/scripts/export_mae_psma_fewshot50_9fold.py" \
    --n-shot "${n}" --out-dir "${dir}" --seed 42
}

_run_nnunet_fewshot() {
  local n="$1"
  local stage="psma_fs${n}_f258"
  local split_dir; split_dir="$(_split_dir "${n}")"
  _ensure_splits "${n}"

  export TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}"
  export TASK1_FEWSHOT_SPLIT_DIR="${split_dir}"
  export TASK1_UDA_FDG_STAMP="${FDG_STAMP}" TASK1_UDA_FDG_BEST="${FDG_BEST}"
  export TASK1_NUM_EPOCHS="${PSMA_EP}" TASK1_TRAIN_ITERS_PER_EPOCH="${PSMA_TR}"
  export TASK1_FS50_VAL_ITERS="${PSMA_VAL}" TASK1_FS50_VAL_EVERY_N_EPOCHS="${PSMA_EVERY}"
  export TASK1_VAL_EVERY_N_EPOCHS="${PSMA_EVERY}" TASK1_VAL_ITERS_PER_EPOCH="${PSMA_VAL}"
  export TASK1_FIXED_BATCH_3D_FULLRES="${PSMA_BS}" TASK1_BEST_BY=val_loss TASK1_VAL_LOSS_ONLY=1
  export TASK1_FOLDS=2,5,8 TASK1_FOLD_GPUS=2:0,5:1,8:3 TASK1_SKIP_TEST20_AT_END=1
  unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

  bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh"
  local parent meta
  meta="$(ls -1t "${VIS}"/iclr2026_nnunet_psma_fs"${n}"_f258_*.txt 2>/dev/null | head -1 || true)"
  [[ -n "${meta}" && -f "${meta}" ]] && parent="$(grep '^PARENT=' "${meta}" | head -1 | cut -d= -f2-)"
  [[ -n "${parent:-}" ]] || parent="$(ls -1dt "${WORK}/nnUNet_results/"*_iclr2026_nnunet_psma_fs"${n}"_f258_* 2>/dev/null | head -1 | xargs -I{} basename {} || true)"
  NN_PARENT="${parent}" TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}" \
    TASK1_FEWSHOT_SPLIT_DIR="${split_dir}" TASK1_UDA_FDG_STAMP="${FDG_STAMP}" TASK1_UDA_FDG_BEST="${FDG_BEST}" \
    bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fewshot_f258_decline_and_test_bg.sh"
}

_run_mae_fewshot() {
  local n="$1"
  local stage="psma_fs${n}_f258"
  local split_dir; split_dir="$(_split_dir "${n}")"
  _ensure_splits "${n}"

  export TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}"
  export TASK1_FEWSHOT_SPLIT_DIR="${split_dir}"
  export TASK1_MAE_FDG_SEG_CKPT="${MAE_FDG_SEG}"
  export TASK1_MAE_NUM_EPOCHS="${PSMA_EP}" TASK1_MAE_BATCH_SIZE="${PSMA_BS}"
  unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

  echo "[pipeline] === MAE fs${n} ==="
  bash "${ROOT}/ICLR2026/run/run_mae_psma_fewshot50_f258_from_fdg_seg_bg.sh"
  local stamp meta
  meta="$(ls -1t "${VIS}"/iclr2026_mae_psma_fdgseg_f258_*.txt 2>/dev/null | head -1 || true)"
  if [[ -n "${meta}" && -f "${meta}" ]]; then
    stamp="$(grep '^STAMP=' "${meta}" | head -1 | cut -d= -f2-)"
  fi
  [[ -n "${stamp:-}" ]] || stamp="$(ls -1dt "${REPO}/runs/"*_iclr2026_mae_psma_fs"${n}"_from_fdg_seg_f258_* 2>/dev/null | head -1 | xargs -I{} basename {} || true)"
  [[ -n "${stamp}" ]] || { echo "[error] MAE stamp missing fs${n}" >&2; exit 1; }

  METHOD=mae STAMP="${stamp}" TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}" \
    bash "${ROOT}/ICLR2026/run/run_eval_psma_test20_f258_bg.sh"
  echo "${stamp}" > "${VIS}/mae_psma_fs${n}_f258_LAST_STAMP.txt"
}

_run_monai_fewshot() {
  local n="$1"
  local stage="psma_fs${n}_f258"
  local split_dir; split_dir="$(_split_dir "${n}")"
  _ensure_splits "${n}"

  export TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}"
  export TASK1_FEWSHOT_SPLIT_DIR="${split_dir}"
  export TASK1_MONAI_FDG_SEG_CKPT="${MONAI_FDG_SEG}"
  export TASK1_MAE_NUM_EPOCHS="${PSMA_EP}" TASK1_MAE_BATCH_SIZE="${PSMA_BS}"
  unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

  echo "[pipeline] === MONAI fs${n} ==="
  bash "${ROOT}/ICLR2026/run/run_monai_psma_fewshot50_f258_from_fdg_seg_bg.sh"
  local stamp="${TASK1_NNUNET_RESULTS_STAMP_NAME:-}"
  [[ -n "${stamp}" ]] || stamp="$(ls -1dt "${REPO}/runs/"*_iclr2026_monai_psma_fs"${n}"_from_fdg_seg_f258_* 2>/dev/null | head -1 | xargs -I{} basename {} || true)"
  [[ -n "${stamp}" ]] || { echo "[error] MONAI stamp missing fs${n}" >&2; exit 1; }

  METHOD=monai STAMP="${stamp}" TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}" \
    bash "${ROOT}/ICLR2026/run/run_eval_psma_test20_f258_bg.sh"
  echo "${stamp}" > "${VIS}/monai_psma_fs${n}_f258_LAST_STAMP.txt"
}

_run_dpdnet_fewshot() {
  local n="$1"
  local stage="psma_fs${n}_f258"
  local split_dir; split_dir="$(_split_dir "${n}")"
  _ensure_splits "${n}"

  # Re-resolve at call time (pipeline may have started before board/script FDG fix).
  local dpd_stamp dpd_best
  dpd_stamp="${TASK1_DPDNET_FDG_STAMP:-20260817_165250_iclr2026_dpdnet_fdg_2gpu_bs3_gbs6_n6_tr70_val0_169ep_gpu01}"
  dpd_best="${TASK1_DPDNET_FDG_BEST:-/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/20260817_165250_iclr2026_dpdnet_fdg_2gpu_bs3_gbs6_n6_tr70_val0_169ep_gpu01/Dataset239_DpDNet_FDG_2ch/STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth}"
  if [[ -z "${TASK1_DPDNET_FDG_STAMP:-}" && -f "${VIS}/dpdnet_fdg_LAST_STAMP.txt" ]]; then
    dpd_stamp="$(tr -d '[:space:]' < "${VIS}/dpdnet_fdg_LAST_STAMP.txt")"
  fi
  if [[ -z "${TASK1_DPDNET_FDG_BEST:-}" && -n "${dpd_stamp}" ]]; then
    local _fdg_fold="${WORK}/nnUNet_results/${dpd_stamp}/Dataset239_DpDNet_FDG_2ch/STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres/fold_0"
    for _c in checkpoint_final.pth checkpoint_latest.pth checkpoint_best.pth; do
      [[ -f "${_fdg_fold}/${_c}" ]] && { dpd_best="${_fdg_fold}/${_c}"; break; }
    done
  fi
  echo "[pipeline] DpDNet fs${n} FDG init=${dpd_best}"

  export TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}"
  export TASK1_FEWSHOT_SPLIT_DIR="${split_dir}"
  export TASK1_DPDNET_FDG_STAMP="${dpd_stamp}" TASK1_DPDNET_FDG_BEST="${dpd_best}"
  export TASK1_DPDNET_NUM_EPOCHS="${PSMA_EP}" TASK1_DPDNET_TRAIN_ITERS="${PSMA_TR}"
  export TASK1_DPDNET_VAL_ITERS="${PSMA_VAL}" TASK1_DPDNET_VAL_EVERY="${PSMA_EVERY}"
  export TASK1_DPDNET_BATCH_SIZE="${PSMA_BS}" TASK1_BEST_BY=val_loss
  export TASK1_FOLDS=2,5,8 TASK1_FOLD_GPUS=2:0,5:1,8:3 TASK1_SKIP_TEST20_AT_END=1
  unset TASK1_NNUNET_RESULTS_STAMP_NAME TASK1_DPDNET_SKIP_PREPARE || true

  echo "[pipeline] === DpDNet fs${n} ==="
  bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_fewshot50_f258_1gpu_bs2_100ep_bg.sh"
  local parent
  parent="$(grep '^PARENT=' "${VIS}"/iclr2026_dpdnet_psma_fs"${n}"_f258_*.txt 2>/dev/null | tail -1 | cut -d= -f2- || true)"
  [[ -n "${parent}" ]] || parent="$(ls -1dt "${WORK}/nnUNet_results/"*_iclr2026_dpdnet_psma_fs"${n}"_f258_* 2>/dev/null | head -1 | xargs -I{} basename {} || true)"
  [[ -n "${parent}" ]] || { echo "[error] DpDNet PARENT missing fs${n}" >&2; exit 1; }

  DPD_PARENT="${parent}" TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}" \
    bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_fewshot_f258_decline_and_test_bg.sh"
  echo "${parent}" > "${VIS}/dpdnet_psma_fs${n}_f258_LAST_STAMP.txt"
}

_run_seganypet_fewshot() {
  local n="$1"
  local stage="psma_fs${n}_f258"
  local split_dir data_root
  split_dir="$(_split_dir "${n}")"
  data_root="${DATA}/task1_train_workspace/seganypet_fewshot${n}_f258"
  _ensure_splits "${n}"

  export TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}"
  export TASK1_FEWSHOT_SPLIT_DIR="${split_dir}"
  export TASK1_SEGANY_DATA_ROOT="${data_root}"
  export TASK1_SEGANY_CKPT="${SEG_FDG_BEST}"
  export TASK1_SEGANY_EPOCHS="${PSMA_EP}" TASK1_SEGANY_BATCH_SIZE="${PSMA_BS}"
  unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

  echo "[pipeline] === SegAnyPET fs${n} ==="
  bash "${ROOT}/ICLR2026/run/run_seganypet_fewshot50_f258_bg.sh"
  local stamp meta
  meta="$(ls -1t "${VIS}"/iclr2026_seganypet_f258_*.txt 2>/dev/null | head -1 || true)"
  if [[ -n "${meta}" && -f "${meta}" ]]; then
    stamp="$(grep '^STAMP=' "${meta}" | head -1 | cut -d= -f2-)"
  fi
  [[ -n "${stamp:-}" ]] || stamp="$(ls -1dt "${REPO}/runs/"*_iclr2026_seganypet_fs"${n}"_f258_* 2>/dev/null | head -1 | xargs -I{} basename {} || true)"
  [[ -n "${stamp}" ]] || { echo "[error] SegAnyPET stamp missing fs${n}" >&2; exit 1; }

  SEG_STAMP="${stamp}" TASK1_FEWSHOT_N="${n}" TASK1_PSMA_BOARD_STAGE="${stage}" \
    TASK1_SEGANY_DATA_ROOT="${data_root}" TASK1_SEGANY_CKPT="${SEG_FDG_BEST}" \
    bash "${ROOT}/ICLR2026/run/run_seganypet_psma_fewshot_f258_decline_and_test_bg.sh"
  echo "${stamp}" > "${VIS}/seganypet_psma_fs${n}_f258_LAST_STAMP.txt"
}

IFS=',' read -r -a _fs <<< "${FEWSHOT_LIST}"
for n in "${_fs[@]}"; do
  n="$(echo "${n}" | tr -d '[:space:]')"
  [[ -n "${n}" ]] || continue
  bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true

  if [[ "${METHODS}" == *mae* || "${METHODS}" == "all" ]]; then _run_mae_fewshot "${n}"; fi
  if [[ "${METHODS}" == *monai* || "${METHODS}" == "all" ]]; then _run_monai_fewshot "${n}"; fi
  if [[ "${METHODS}" == *dpdnet* || "${METHODS}" == "all" ]]; then _run_dpdnet_fewshot "${n}"; fi
  if [[ "${METHODS}" == *seganypet* || "${METHODS}" == "all" ]]; then _run_seganypet_fewshot "${n}"; fi
  if [[ "${METHODS}" == *nnunet* || "${METHODS}" == "all" ]]; then _run_nnunet_fewshot "${n}"; fi
done

if [[ "${METHODS}" != *nnunet* && "${METHODS}" != "all" ]]; then
  echo "status=ok methods=${METHODS} fs=${FEWSHOT_LIST} at=$(TZ=Asia/Shanghai date '+%F %T %Z')" \
    > "${VIS}/TASK1_FS10_FS5_METHODS_PIPELINE_DONE.txt"
fi

if [[ "${TASK1_NNUNET_RERUN_AT_END:-0}" == "1" ]]; then
  echo "[pipeline] === nnUNet fs10/fs5 rerun (FDG169 final) ==="
  for n in "${_fs[@]}"; do
    n="$(echo "${n}" | tr -d '[:space:]')"
    [[ -n "${n}" ]] || continue
    bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
    _run_nnunet_fewshot "${n}"
  done
fi

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"queue\":[],\"updated_note\":\"PSMA fs10/fs5 pipeline done · methods=${METHODS}\"}" || true

echo "[pipeline] ALL DONE fs=${FEWSHOT_LIST} methods=${METHODS}"
