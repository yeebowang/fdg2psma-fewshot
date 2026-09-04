#!/usr/bin/env bash
# Continue competition *_scratch after fs50: PSMA fs10 → fs5 → TEST20 (reuse each method's FDG ckpt).
# Does not touch Dataset619 / pretrained rows; does not revive queue_keeper.
#
#   bash ICLR2026/run/run_competition_scratch_fs10_fs5_continue_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
LOG="${VIS}/nohup_competition_scratch_fs10_fs5_continue.log"
mkdir -p "${VIS}"
exec > >(tee -a "${LOG}") 2>&1

METHODS=(${TASK1_COMP_SCRATCH_METHODS:-hemingduo_scratch chenyixin_scratch})
FEWSHOT_LIST="${TASK1_FEWSHOT_LIST:-10,5}"
PSMA_EP="${TASK1_NUM_EPOCHS:-100}"
PSMA_TR="${TASK1_TRAIN_ITERS_PER_EPOCH:-25}"
PSMA_VAL="${TASK1_FS50_VAL_ITERS:-25}"
PSMA_EVERY="${TASK1_FS50_VAL_EVERY_N_EPOCHS:-20}"
PSMA_BS="${TASK1_FIXED_BATCH_3D_FULLRES:-2}"

echo "[comp-scratch-cont] $(date '+%F %T') methods=${METHODS[*]} few=${FEWSHOT_LIST}"
python3 "${ROOT}/ICLR2026/scripts/assert_competition_board_weights.py" || exit 1

_resolve_fdg() {
  local method="$1"
  local stamp ckpt fold
  stamp="$(tr -d '[:space:]' < "${VIS}/${method}_fdg_LAST_STAMP.txt" 2>/dev/null || true)"
  [[ -n "${stamp}" ]] || {
    stamp="$(ls -1dt "${WORK}/nnUNet_results/"*_iclr2026_${method}_fdg_*_169ep* 2>/dev/null | head -1 | xargs -I{} basename {} || true)"
  }
  [[ -n "${stamp}" ]] || return 1
  fold="${WORK}/nnUNet_results/${stamp}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0"
  for c in checkpoint_final.pth checkpoint_latest.pth checkpoint_best.pth; do
    if [[ -f "${fold}/${c}" ]]; then
      echo "${stamp}|${fold}/${c}"
      return 0
    fi
  done
  return 1
}

_ensure_splits() {
  local n="$1" dir
  dir="${ROOT}/ICLR2026/data/splits_mae_psma_fewshot${n}_9fold"
  [[ -f "${dir}/fold0_nnunet.json" ]] || python3 "${ROOT}/ICLR2026/scripts/export_mae_psma_fewshot50_9fold.py" \
    --n-shot "${n}" --out-dir "${dir}" --seed 42
}

IFS=',' read -r -a FEWS <<< "${FEWSHOT_LIST}"

for method in "${METHODS[@]}"; do
  pair="$(_resolve_fdg "${method}")" || { echo "[error] no FDG ckpt for ${method}" >&2; exit 1; }
  FDG_STAMP="${pair%%|*}"
  FDG_CKPT="${pair#*|}"
  echo "[comp-scratch-cont] ${method} FDG=${FDG_STAMP} ckpt=${FDG_CKPT}"

  for n in "${FEWS[@]}"; do
    stage="psma_fs${n}_f258"
    split_dir="${ROOT}/ICLR2026/data/splits_mae_psma_fewshot${n}_9fold"
    _ensure_splits "${n}"
    echo "[comp-scratch-cont] === ${method} ${stage} $(date '+%F %T') ==="

    python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
      --board "${BOARD}" --no-plot \
      --patch-json "{\"methods\":{\"${method}\":{\"${stage}\":{\"status\":\"running\",\"note\":\"fs${n} after scratch FDG\",\"bs\":${PSMA_BS},\"total_epochs\":${PSMA_EP},\"train_iters\":${PSMA_TR},\"val_iters\":${PSMA_VAL},\"test_invalidated\":true,\"fold_dice\":{},\"mean\":null}}},\"updated_note\":\"${method} ${stage} running\"}" || true

    export TASK1_BOARD_METHOD="${method}"
    export TASK1_FEWSHOT_N="${n}"
    export TASK1_PSMA_BOARD_STAGE="${stage}"
    export TASK1_FEWSHOT_SPLIT_DIR="${split_dir}"
    export TASK1_UDA_FDG_STAMP="${FDG_STAMP}"
    export TASK1_UDA_FDG_BEST="${FDG_CKPT}"
    export TASK1_NUM_EPOCHS="${PSMA_EP}"
    export TASK1_TRAIN_ITERS_PER_EPOCH="${PSMA_TR}"
    export TASK1_FS50_VAL_ITERS="${PSMA_VAL}"
    export TASK1_FS50_VAL_EVERY_N_EPOCHS="${PSMA_EVERY}"
    export TASK1_VAL_EVERY_N_EPOCHS="${PSMA_EVERY}"
    export TASK1_VAL_ITERS_PER_EPOCH="${PSMA_VAL}"
    export TASK1_FIXED_BATCH_3D_FULLRES="${PSMA_BS}"
    export TASK1_BEST_BY=val_loss
    export TASK1_VAL_LOSS_ONLY=1
    export TASK1_FOLDS=2,5,8
    export TASK1_FOLD_GPUS=2:0,5:1,8:3
    export TASK1_TEST_SKIP_DONE=1
    unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

    bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh"
    echo "[comp-scratch-cont] ${method} ${stage} 3fold DONE $(date '+%F %T')"
  done
done

echo "[comp-scratch-cont] 3fold cascade done → hand off to 9fold extra (if not already running)"
if ! pgrep -f 'run_competition_scratch_extra_folds_9fold_after_3fold_bg.sh' >/dev/null 2>&1; then
  nohup bash "${ROOT}/ICLR2026/run/run_competition_scratch_extra_folds_9fold_after_3fold_bg.sh" \
    >> "${VIS}/nohup_competition_scratch_9fold_launcher.log" 2>&1 &
  echo "[comp-scratch-cont] launched 9fold pid=$!"
fi

python3 "${ROOT}/ICLR2026/scripts/backfill_competition_fp_fn_from_score_detail.py" || true
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" \
  --png "${VIS}/progress_iclr2026_aligned_fdg_fs50_f258_board.png" \
  --patch-json '{"updated_note":"competition scratch 3fold continue done; 9fold queued"}' || true
echo "[comp-scratch-cont] ALL DONE $(date '+%F %T')"
