#!/usr/bin/env bash
# After competition *_scratch 3-fold (2/5/8) stages finish: train extra folds → full 9-fold TEST20.
# Waits for the fs10/fs5 continue queue if still running, then processes fs50/fs10/fs5.
#
#   bash ICLR2026/run/run_competition_scratch_extra_folds_9fold_after_3fold_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
LOG="${VIS}/nohup_competition_scratch_extra_folds_9fold.log"
mkdir -p "${VIS}"
exec > >(tee -a "${LOG}") 2>&1

METHODS=(${TASK1_COMP_SCRATCH_METHODS:-hemingduo_scratch chenyixin_scratch})
FEWSHOT_LIST="${TASK1_FEWSHOT_LIST:-50,10,5}"
EXTRA_FOLDS="${TASK1_EXTRA_FOLDS:-0,1,3,4,6,7}"
ALL_FOLDS="${TASK1_ALL_FOLDS:-0,1,2,3,4,5,6,7,8}"
EXTRA_FOLD_GPUS="${TASK1_EXTRA_FOLD_GPUS:-0:0,1:1,3:3,4:0,6:1,7:3}"
# One fold per GPU when launching all 9 in parallel (old map put 0/2/4/7 on GPU0 → OOM).
ALL_FOLD_GPUS="${TASK1_ALL_FOLD_GPUS:-0:0,1:1,2:3,3:0,4:1,5:3,6:0,7:1,8:3}"
PSMA_EP="${TASK1_NUM_EPOCHS:-100}"
PSMA_TR="${TASK1_TRAIN_ITERS_PER_EPOCH:-25}"
PSMA_VAL="${TASK1_FS50_VAL_ITERS:-25}"
PSMA_EVERY="${TASK1_FS50_VAL_EVERY_N_EPOCHS:-20}"
PSMA_BS="${TASK1_FIXED_BATCH_3D_FULLRES:-2}"

echo "[comp-9fold] $(date '+%F %T') methods=${METHODS[*]} few=${FEWSHOT_LIST} extra=${EXTRA_FOLDS}"

# Wait for 3-fold continue if live (do not steal GPUs mid fs10)
_continue_live() {
  pgrep -af 'run_competition_scratch_fs10_fs5_continue_bg.sh' 2>/dev/null \
    | grep -v 'extra_folds_9fold' | grep -v 'pgrep' | grep -vq 'grep' || return 1
  return 0
}
while _continue_live; do
  echo "[comp-9fold] wait fs10/fs5 continue… $(date '+%F %T')"
  sleep 60
done
echo "[comp-9fold] continue cleared → start 9fold $(date '+%F %T')"

python3 "${ROOT}/ICLR2026/scripts/assert_competition_board_weights.py" || exit 1

_resolve_fdg() {
  local method="$1" stamp fold c
  stamp="$(tr -d '[:space:]' < "${VIS}/${method}_fdg_LAST_STAMP.txt" 2>/dev/null || true)"
  [[ -n "${stamp}" ]] || stamp="$(ls -1dt "${WORK}/nnUNet_results/"*_iclr2026_${method}_fdg_*_169ep* 2>/dev/null | head -1 | xargs -I{} basename {} || true)"
  [[ -n "${stamp}" ]] || return 1
  fold="${WORK}/nnUNet_results/${stamp}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0"
  for c in checkpoint_final.pth checkpoint_latest.pth checkpoint_best.pth; do
    [[ -f "${fold}/${c}" ]] && { echo "${stamp}|${fold}/${c}"; return 0; }
  done
  return 1
}

_board_stamp() {
  local method="$1" stage="$2"
  python3 - <<PY
import json
from pathlib import Path
b = json.loads(Path("${BOARD}").read_text())
m = (b.get("methods") or {}).get("${method}") or {}
st = m.get("${stage}") or {}
print((st.get("stamp") or "").strip())
PY
}

_ensure_splits() {
  local n="$1" dir
  dir="${ROOT}/ICLR2026/data/splits_mae_psma_fewshot${n}_9fold"
  [[ -f "${dir}/fold0_nnunet.json" ]] || python3 "${ROOT}/ICLR2026/scripts/export_mae_psma_fewshot50_9fold.py" \
    --n-shot "${n}" --out-dir "${dir}" --seed 42
}

_fold_trained() {
  local parent="$1" fold="$2"
  local fd="${WORK}/nnUNet_results/${parent}_f${fold}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0"
  [[ -f "${fd}/checkpoint_final.pth" || -f "${fd}/checkpoint_best.pth" ]]
}

IFS=',' read -r -a FEWS <<< "${FEWSHOT_LIST}"

for method in "${METHODS[@]}"; do
  pair="$(_resolve_fdg "${method}")" || { echo "[error] no FDG for ${method}" >&2; exit 1; }
  FDG_STAMP="${pair%%|*}"
  FDG_CKPT="${pair#*|}"
  echo "[comp-9fold] ${method} FDG=${FDG_STAMP}"

  for n in "${FEWS[@]}"; do
    stage="psma_fs${n}_f258"
    parent="$(_board_stamp "${method}" "${stage}")"
    if [[ -z "${parent}" ]]; then
      echo "[comp-9fold] skip ${method}.${stage}: no stamp yet"
      continue
    fi
    split_dir="${ROOT}/ICLR2026/data/splits_mae_psma_fewshot${n}_9fold"
    _ensure_splits "${n}"

    need=0
    IFS=',' read -r -a EX <<< "${EXTRA_FOLDS}"
    for f in "${EX[@]}"; do
      _fold_trained "${parent}" "${f}" || need=1
    done
    if [[ "${need}" == "0" ]]; then
      echo "[comp-9fold] ${method}.${stage} extra folds already trained → re-TEST20 9fold"
    else
      echo "[comp-9fold] === ${method} ${stage} train extra ${EXTRA_FOLDS} $(date '+%F %T') ==="
      python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
        --board "${BOARD}" --no-plot \
        --patch-json "{\"methods\":{\"${method}\":{\"${stage}\":{\"status\":\"running\",\"note\":\"9fold extra · folds ${EXTRA_FOLDS}\"}}},\"updated_note\":\"${method} ${stage} 9fold extra\"}" || true

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
      export TASK1_FOLDS="${EXTRA_FOLDS}"
      export TASK1_FOLD_GPUS="${EXTRA_FOLD_GPUS}"
      export TASK1_SKIP_TEST20_AT_END=1
      export TASK1_NNUNET_RESULTS_STAMP_NAME="${parent}"

      bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh"
    fi

    echo "[comp-9fold] ${method}.${stage} TEST20 all folds ${ALL_FOLDS}"
    export PARENT_STAMP="${parent}"
    export TASK1_NNUNET_RESULTS_STAMP_NAME="${parent}"
    export TASK1_BOARD_METHOD="${method}"
    export TASK1_FEWSHOT_N="${n}"
    export TASK1_PSMA_BOARD_STAGE="${stage}"
    export TASK1_FOLDS="${ALL_FOLDS}"
    export TASK1_FOLD_GPUS="${ALL_FOLD_GPUS}"
    # Default skip completed score_detail folds; set TASK1_TEST_SKIP_DONE=0 to force re-eval.
    export TASK1_TEST_SKIP_DONE="${TASK1_TEST_SKIP_DONE:-1}"
    # Keep ≤1 predict/GPU for --save_probabilities (5 concurrent shards OOMed on 3090).
    export TASK1_UDA_PRED_PER_GPU="${TASK1_UDA_PRED_PER_GPU:-1}"
    bash "${ROOT}/ICLR2026/run/run_nnunet_psma_test20_f258_parallel.sh"

    python3 "${ROOT}/ICLR2026/scripts/backfill_competition_fp_fn_from_score_detail.py" || true
    python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
      --board "${BOARD}" --no-plot \
      --patch-json "{\"updated_note\":\"${method} ${stage} 9fold TEST20 refreshed\"}" || true
  done
done

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" \
  --png "${VIS}/progress_iclr2026_aligned_fdg_fs50_f258_board.png" \
  --patch-json '{"updated_note":"competition scratch 9fold extra done"}' || true
echo "[comp-9fold] ALL DONE $(date '+%F %T')"
