#!/usr/bin/env bash
# Continue DpDNet dual-enc after fs50 DONE:
#   3GPU fs10 → 3GPU fs5 → 1GPU×3 (fc70 / PSMA fs0 / FDG TEST)
# Does not restart full queue_keeper. GPU2 untouched.
#
#   bash ICLR2026/run/run_dpdnet_dualenc_continue_after_fs50_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
PID_FILE="${VIS}/dpdnet_dualenc_continue_after_fs50.pid"
LOG="${VIS}/nohup_dpdnet_dualenc_continue_after_fs50.log"

FDG_STAMP="${TASK1_DPDNET_FDG_STAMP:-$(tr -d '[:space:]' < "${VIS}/dpdnet_dualenc_fdg_LAST_STAMP.txt" 2>/dev/null || true)}"
[[ -n "${FDG_STAMP}" ]] || FDG_STAMP="20260831_203650_iclr2026_dpdnet_dualenc_fdg_1gpu_bs6_n6_tr70_val0_169ep_gpu0"
FDG_TF="STUNetTrainer_small_prompt_pretrain__nnUNetPlans__3d_fullres"
FDG_FOLD="${WORK}/nnUNet_results/${FDG_STAMP}/Dataset239_DpDNet_FDG_2ch/${FDG_TF}/fold_0"
FDG_CKPT=""
for c in checkpoint_final.pth checkpoint_best.pth checkpoint_latest.pth; do
  [[ -f "${FDG_FOLD}/${c}" ]] && { FDG_CKPT="${FDG_FOLD}/${c}"; break; }
done
[[ -n "${FDG_CKPT}" && -f "${FDG_CKPT}" ]] || { echo "[error] dualenc FDG ckpt missing under ${FDG_FOLD}" >&2; exit 1; }

mkdir -p "${VIS}"
if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" && "${old}" != "$$" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[dualenc-cont] already running pid=${old}"
    exit 0
  fi
fi
echo $$ > "${PID_FILE}"
exec > >(tee -a "${LOG}") 2>&1
echo "[dualenc-cont] $(date '+%F %T') pid=$$ FDG=${FDG_STAMP} ckpt=${FDG_CKPT}"

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD}"
export TASK1_BOARD_METHOD=dpdnet_dualenc
export TRAINER=STUNetTrainer_small_prompt_pretrain
export TASK1_DPDNET_SKIP_ENCODER_INIT=1
export TASK1_DPDNET_FDG_STAMP="${FDG_STAMP}"
export TASK1_DPDNET_FDG_BEST="${FDG_CKPT}"
export TASK1_DPDNET_FDG_FORCE_STAMP=1
export TASK1_DPDNET_FDG_TF="${FDG_TF}"
export TASK1_DPDNET_NUM_EPOCHS=100
export TASK1_DPDNET_TRAIN_ITERS=25
export TASK1_DPDNET_VAL_ITERS=25
export TASK1_DPDNET_VAL_EVERY=20
export TASK1_DPDNET_BATCH_SIZE=2
export TASK1_BEST_BY=val_loss
export TASK1_FOLDS=2,5,8
export TASK1_FOLD_GPUS=2:0,5:1,8:3
export TASK1_TEST_SKIP_DONE=1
export TASK1_UDA_PRED_PER_GPU="${TASK1_UDA_PRED_PER_GPU:-5}"

_board() {
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD}" --no-plot --patch-json "$1" || true
}

_stage_done() {
  local stage="$1"
  python3 - <<PY
import json
from pathlib import Path
b=json.loads(Path("${BOARD}").read_text())
s=(b.get("methods") or {}).get("dpdnet_dualenc", {}).get("${stage}") or {}
st=(s.get("status") or "").lower()
mean=s.get("mean")
folds=s.get("fold_dice") or {}
ok = st == "done" and mean is not None and (len(folds) >= 3 or s.get("training_free") is True or s.get("eval_done"))
# board lag: stamp + aggregate_test20 on disk also counts as done
if not ok:
    stamp = (s.get("stamp") or "").strip()
    if stamp:
        agg = Path("${WORK}") / "nnUNet_results" / stamp / "aggregate_test20_dice_f258.json"
        if agg.is_file():
            try:
                ad = json.loads(agg.read_text())
                if (ad.get("folds") or ad.get("fold_mean") is not None):
                    ok = True
            except Exception:
                pass
print("1" if ok else "0")
PY
}

_run_fewshot() {
  local n="$1"
  local stage="psma_fs${n}_f258"
  if [[ "$(_stage_done "${stage}")" == "1" ]]; then
    echo "[dualenc-cont] skip fs${n} (already done on board)"
    return 0
  fi
  local split_dir="${CTRL}/ICLR2026/data/splits_mae_psma_fewshot${n}_9fold"
  [[ -f "${split_dir}/fold0_nnunet.json" ]] || python3 "${CTRL}/ICLR2026/scripts/export_mae_psma_fewshot50_9fold.py" \
    --n-shot "${n}" --out-dir "${split_dir}" --seed 42

  echo "[dualenc-cont] === 3GPU dualenc fs${n} ==="
  unset TASK1_NNUNET_RESULTS_STAMP_NAME || true
  # disarm needs a stamp; use FDG stamp so any leftover arm for prior stage is cleared
  TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}" \
    bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
  _board "{\"methods\":{\"dpdnet_dualenc\":{\"${stage}\":{\"status\":\"running\",\"note\":\"3GPU · dualenc fs${n} f258 · tr25/val25\"}}},\"updated_note\":\"dualenc 3GPU fs${n}\"}"

  TASK1_FEWSHOT_N="${n}" \
  TASK1_PSMA_BOARD_STAGE="${stage}" \
  TASK1_FEWSHOT_SPLIT_DIR="${split_dir}" \
  TASK1_DPDNET_SKIP_PREPARE=0 \
  TASK1_SKIP_TEST20_AT_END=0 \
    bash "${CTRL}/ICLR2026/run/run_dpdnet_psma_fewshot50_f258_1gpu_bs2_100ep_bg.sh"

  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true
}

# --- Phase A: 3GPU fs10 then fs5 ---
_run_fewshot 10
_run_fewshot 5

# --- Phase B: 1GPU×3 ---
echo "[dualenc-cont] === 1GPU×3 fc70 / fs0 / FDG TEST ==="
_board '{"updated_note":"dualenc 1GPU×3 · fc70 / fs0 / FDG TEST"}'
TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}" \
  bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

# fc70 on GPU0
if [[ "$(_stage_done psma_fc70)" != "1" ]]; then
  TASK1_BOARD_METHOD=dpdnet_dualenc \
  TASK1_DPDNET_FDG_BEST="${FDG_CKPT}" \
  TASK1_DPDNET_FDG_STAMP="${FDG_STAMP}" \
  TRAINER=STUNetTrainer_small_prompt_pretrain \
  TASK1_DPDNET_SKIP_ENCODER_INIT=1 \
  TASK1_PSMA_FC70_GPU=0 \
    bash "${CTRL}/ICLR2026/run/run_dpdnet_psma_fc70_decline_and_test_bg.sh" &
  p_fc70=$!
else
  p_fc70=""
  echo "[dualenc-cont] skip fc70"
fi

# PSMA fs0 on GPU1
if [[ "$(_stage_done psma_fs0)" != "1" ]]; then
  METHOD=dpdnet_dualenc TASK1_TEST_SKIP_DONE=0 TASK1_CUDA_VISIBLE_DEVICES=1 \
  TASK1_DPDNET_FDG_BEST="${FDG_CKPT}" TASK1_DPDNET_FDG_STAMP="${FDG_STAMP}" \
    bash "${CTRL}/ICLR2026/run/run_eval_fdg_shared_test20_bg.sh" &
  p_fs0=$!
else
  p_fs0=""
  echo "[dualenc-cont] skip fs0"
fi

# FDG TEST on GPU3
if [[ "$(_stage_done fdg_test20)" != "1" ]]; then
  METHOD=dpdnet_dualenc TASK1_TEST_SKIP_DONE=0 TASK1_CUDA_VISIBLE_DEVICES=3 \
  TASK1_DPDNET_FDG_BEST="${FDG_CKPT}" TASK1_DPDNET_FDG_STAMP="${FDG_STAMP}" \
    bash "${CTRL}/ICLR2026/run/run_eval_fdg_test20_bg.sh" &
  p_fdg=$!
else
  p_fdg=""
  echo "[dualenc-cont] skip FDG TEST"
fi

[[ -n "${p_fs0}" ]] && { wait "${p_fs0}" || echo "[dualenc-cont] fs0 rc=$?"; }
[[ -n "${p_fdg}" ]] && { wait "${p_fdg}" || echo "[dualenc-cont] fdg rc=$?"; }
[[ -n "${p_fc70}" ]] && { wait "${p_fc70}" || echo "[dualenc-cont] fc70 rc=$?"; }

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true
{
  echo "done_at=$(TZ=Asia/Shanghai date '+%F %T %Z')"
  echo "status=ok"
} > "${VIS}/TASK1_DPDNET_DUALENC_CONTINUE_DONE.txt"
echo "[dualenc-cont] ALL DONE $(date '+%F %T')"
