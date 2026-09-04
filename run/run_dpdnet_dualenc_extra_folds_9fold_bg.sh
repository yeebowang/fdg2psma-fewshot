#!/usr/bin/env bash
# DpDNet dual-enc: fill missing folds 0,1,3,4,6,7 for fs50/10/5 → full 9-fold + TEST20.
# Reuses board PARENT stamps. GPU2 untouched. Cascade: fs50 → fs10 → fs5.
#
#   bash ICLR2026/run/run_dpdnet_dualenc_extra_folds_9fold_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
PID_FILE="${VIS}/dpdnet_dualenc_extra_folds_9fold.pid"
LOG="${VIS}/nohup_dpdnet_dualenc_extra_folds_9fold.log"

EXTRA_FOLDS="${TASK1_EXTRA_FOLDS:-0,1,3,4,6,7}"
ALL_FOLDS="${TASK1_ALL_FOLDS:-0,1,2,3,4,5,6,7,8}"
EXTRA_FOLD_GPUS="${TASK1_EXTRA_FOLD_GPUS:-0:0,1:1,3:3,4:0,6:1,7:3}"
ALL_FOLD_GPUS="${TASK1_ALL_FOLD_GPUS:-0:0,1:1,2:0,3:3,4:0,5:1,6:3,7:0,8:3}"
FEWSHOT_LIST="${TASK1_FEWSHOT_LIST:-50,10,5}"

FDG_STAMP="${TASK1_DPDNET_FDG_STAMP:-$(tr -d '[:space:]' < "${VIS}/dpdnet_dualenc_fdg_LAST_STAMP.txt" 2>/dev/null || true)}"
[[ -n "${FDG_STAMP}" ]] || FDG_STAMP="20260831_203650_iclr2026_dpdnet_dualenc_fdg_1gpu_bs6_n6_tr70_val0_169ep_gpu0"
FDG_TF="STUNetTrainer_small_prompt_pretrain__nnUNetPlans__3d_fullres"
FDG_FOLD="${WORK}/nnUNet_results/${FDG_STAMP}/Dataset239_DpDNet_FDG_2ch/${FDG_TF}/fold_0"
FDG_CKPT=""
for c in checkpoint_final.pth checkpoint_best.pth checkpoint_latest.pth; do
  [[ -f "${FDG_FOLD}/${c}" ]] && { FDG_CKPT="${FDG_FOLD}/${c}"; break; }
done
[[ -n "${FDG_CKPT}" ]] || { echo "[error] dualenc FDG ckpt missing" >&2; exit 1; }

PSMA_EP="${TASK1_DPDNET_NUM_EPOCHS:-100}"
PSMA_TR="${TASK1_DPDNET_TRAIN_ITERS:-25}"
PSMA_VAL="${TASK1_DPDNET_VAL_ITERS:-25}"
PSMA_EVERY="${TASK1_DPDNET_VAL_EVERY:-20}"
PSMA_BS="${TASK1_DPDNET_BATCH_SIZE:-2}"

mkdir -p "${VIS}"
if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" && "${old}" != "$$" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[dualenc-9fold] already running pid=${old}"
    exit 0
  fi
fi
echo $$ > "${PID_FILE}"
exec > >(tee -a "${LOG}") 2>&1
echo "[dualenc-9fold] $(date '+%F %T') pid=$$ FDG=${FDG_STAMP} folds=${EXTRA_FOLDS}"

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD}"
export TASK1_BOARD_METHOD=dpdnet_dualenc
export TRAINER=STUNetTrainer_small_prompt_pretrain
export TASK1_DPDNET_SKIP_ENCODER_INIT=1
export TASK1_DPDNET_FDG_STAMP="${FDG_STAMP}"
export TASK1_DPDNET_FDG_BEST="${FDG_CKPT}"
export TASK1_DPDNET_FDG_FORCE_STAMP=1
export TASK1_DPDNET_FDG_TF="${FDG_TF}"
export TASK1_UDA_PRED_PER_GPU="${TASK1_UDA_PRED_PER_GPU:-5}"

_board_stamp() {
  local stage="$1"
  python3 - <<PY
import json
from pathlib import Path
b=json.loads(Path("${BOARD}").read_text())
st=((b.get("methods") or {}).get("dpdnet_dualenc") or {}).get("${stage}") or {}
print((st.get("stamp") or "") if isinstance(st, dict) else "")
PY
}

_missing_extra() {
  local parent="$1"
  python3 - <<PY
from pathlib import Path
work = Path("${WORK}") / "nnUNet_results"
parent = "${parent}"
extra = [int(x) for x in "${EXTRA_FOLDS}".split(",") if x.strip() != ""]
miss = []
for fold in extra:
    ok = False
    for cand in (work / f"{parent}_f{fold}", work / parent):
        if any(cand.glob(f"**/fold_{fold}/checkpoint_*.pth")):
            ok = True
            break
        if cand.name.endswith(f"_f{fold}") and any(cand.glob("**/checkpoint_*.pth")):
            ok = True
            break
    if not ok:
        miss.append(str(fold))
print(",".join(miss))
PY
}

_run_one() {
  local n="$1"
  local stage="psma_fs${n}_f258"
  local split_dir parent miss
  split_dir="${ROOT}/ICLR2026/data/splits_mae_psma_fewshot${n}_9fold"
  [[ -f "${split_dir}/fold0_nnunet.json" ]] || python3 "${ROOT}/ICLR2026/scripts/export_mae_psma_fewshot50_9fold.py" \
    --n-shot "${n}" --out-dir "${split_dir}" --seed 42
  parent="$(_board_stamp "${stage}")"
  [[ -n "${parent}" ]] || { echo "[error] dualenc stamp missing for ${stage}" >&2; return 1; }
  miss="$(_missing_extra "${parent}")"
  if [[ -z "${miss}" ]]; then
    echo "[dualenc-9fold] fs${n} extra folds already present → TEST20 only"
  else
    echo "[dualenc-9fold] === fs${n} train missing folds ${miss} PARENT=${parent} ==="
    python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
      --board "${BOARD}" --no-plot \
      --patch-json "{\"methods\":{\"dpdnet_dualenc\":{\"${stage}\":{\"status\":\"running\",\"stamp\":\"${parent}\",\"note\":\"9fold extra · dualenc fs${n} · folds ${miss}\"}}},\"updated_note\":\"dualenc 9fold extra fs${n}\"}" || true

    TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}" \
      bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true

    # re-prepare Dataset240 for this fewshot N (fc70/other overwrote splits)
    TASK1_FEWSHOT_N="${n}" \
    TASK1_PSMA_BOARD_STAGE="${stage}" \
    TASK1_FEWSHOT_SPLIT_DIR="${split_dir}" \
    TASK1_DPDNET_NUM_EPOCHS="${PSMA_EP}" \
    TASK1_DPDNET_TRAIN_ITERS="${PSMA_TR}" \
    TASK1_DPDNET_VAL_ITERS="${PSMA_VAL}" \
    TASK1_DPDNET_VAL_EVERY="${PSMA_EVERY}" \
    TASK1_DPDNET_BATCH_SIZE="${PSMA_BS}" \
    TASK1_BEST_BY=val_loss \
    TASK1_FOLDS="${miss}" \
    TASK1_FOLD_GPUS="${EXTRA_FOLD_GPUS}" \
    TASK1_SKIP_TEST20_AT_END=1 \
    TASK1_DPDNET_SKIP_PREPARE=0 \
    TASK1_NNUNET_RESULTS_STAMP_NAME="${parent}" \
      bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_fewshot50_f258_1gpu_bs2_100ep_bg.sh"
  fi

  echo "[dualenc-9fold] fs${n} → TEST20 all folds ${ALL_FOLDS}"
  TASK1_NNUNET_RESULTS_STAMP_NAME="${parent}" \
    bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
  PARENT_STAMP="${parent}" \
  TASK1_NNUNET_RESULTS_STAMP_NAME="${parent}" \
  TASK1_FOLDS="${ALL_FOLDS}" \
  TASK1_FOLD_GPUS="${ALL_FOLD_GPUS}" \
  TASK1_TEST_SKIP_DONE=0 \
  TASK1_BOARD_METHOD=dpdnet_dualenc \
  TRAINER=STUNetTrainer_small_prompt_pretrain \
    bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_test20_f258_parallel.sh"

  python3 "${ROOT}/ICLR2026/scripts/backfill_mim_dualenc_fp_fn_from_score_detail.py" --board "${BOARD}" || true
  python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true
}

IFS=',' read -r -a _fs <<< "${FEWSHOT_LIST}"
for n in "${_fs[@]}"; do
  n="$(echo "${n}" | tr -d '[:space:]')"
  [[ -n "${n}" ]] || continue
  _run_one "${n}"
done

python3 "${ROOT}/ICLR2026/scripts/backfill_mim_dualenc_fp_fn_from_score_detail.py" --board "${BOARD}" || true
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true
{
  echo "done_at=$(TZ=Asia/Shanghai date '+%F %T %Z')"
  echo "status=ok"
} > "${VIS}/TASK1_DPDNET_DUALENC_EXTRA_9FOLD_DONE.txt"
echo "[dualenc-9fold] ALL DONE $(date '+%F %T')"
