#!/usr/bin/env bash
# Baseline1 (FDG-only train) → predict + Dice on shared PSMA val (fewshot50 9fold val, 59 cases).
# Zero-shot transfer from FDG; report folds 0..8 (same dice — shared val, frozen model).
set -euo pipefail

CTRL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
LOG_DIR="${CTRL}/ICLR2026/vis"
STAMP_TZ="${STAMP_TZ:-Asia/Shanghai}"
FOLDS_CSV="${TASK1_EVAL_FOLDS:-0,1,2,3,4,5,6,7,8}"

B1_STAMP="${TASK1_BASELINE1_STAMP:-20260810_104431_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val10_3000ep}"
NN_RESULTS="${WORK}/nnUNet_results/${B1_STAMP}"
CKPT="${TASK1_BASELINE1_CKPT:-${NN_RESULTS}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth}"
CASES_JSON="${TASK1_PSMA_VAL_JSON:-${CTRL}/ICLR2026/data/splits_baseline1_psma_val.json}"
# shared val sanity vs any fewshot fold (fold0)
FOLD_JSON="${CTRL}/ICLR2026/data/splits_mae_psma_fewshot50_9fold/fold0_nnunet.json"

STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME:-}"
if [[ -z "${STAMP}" || "${STAMP}" != *baseline1*eval* ]]; then
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_baseline1_fdg_eval_psma_9fold_gpu013"
fi
OUT="${WORK}/nnUNet_results/${STAMP}/psma_val_9fold"
PRED_OUT="${OUT}/predict"
SCORE_JSON="${OUT}/aggregate_val_dice_9fold.json"
# keep legacy filename alias for older plot/chain readers
SCORE_JSON_ALIAS="${OUT}/aggregate_val_dice_f258.json"
VIS_JSON="${LOG_DIR}/aggregate_baseline1_fdg_eval_psma_9fold_${STAMP}.json"
PIPE_LOG="${LOG_DIR}/nohup_baseline1_fdg_eval_psma_9fold_${STAMP}.log"

mkdir -p "${OUT}" "${LOG_DIR}"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[b1-eval-9fold] STAMP=${STAMP}"
echo "[b1-eval-9fold] folds=${FOLDS_CSV}"
echo "[b1-eval-9fold] ckpt=${CKPT}"
echo "[b1-eval-9fold] cases=${CASES_JSON}"

[[ -f "${CKPT}" ]] || { echo "[error] missing ckpt" >&2; exit 1; }
[[ -f "${CASES_JSON}" ]] || { echo "[error] missing cases json" >&2; exit 1; }

# Build a predict JSON with train=val (reuse UDA predict script which reads train)
PRED_CASES="${OUT}/cases_val_as_train.json"
python3 - <<PY
import json
from pathlib import Path
src = Path("${CASES_JSON}")
raw = json.loads(src.read_text(encoding="utf-8"))
if isinstance(raw, list) and raw and isinstance(raw[0], dict):
    cases = list(raw[0].get("val") or raw[0].get("train") or [])
elif isinstance(raw, dict):
    cases = list(raw.get("val") or raw.get("train") or raw.get("cases") or [])
else:
    cases = list(raw)
# sanity: shared val matches fewshot fold0 val
fold = json.loads(Path("${FOLD_JSON}").read_text(encoding="utf-8"))
fold_val = set(fold[0]["val"])
assert set(cases) == fold_val, f"val mismatch vs fold0: {len(cases)} vs {len(fold_val)}"
# also assert all 9 folds share the same val
split_dir = Path("${CTRL}/ICLR2026/data/splits_mae_psma_fewshot50_9fold")
for i in range(9):
    j = json.loads((split_dir / f"fold{i}_nnunet.json").read_text(encoding="utf-8"))
    assert set(j[0]["val"]) == fold_val, f"fold{i} val differs from fold0"
out = Path("${PRED_CASES}")
out.write_text(json.dumps([{"train": cases, "val": []}], indent=2) + "\n", encoding="utf-8")
print(f"[b1-eval-9fold] n_val={len(cases)} shared across folds 0..8 → {out}")
PY

export TASK1_BASE="${DATA}"
export TASK1_UDA_CKPT="${CKPT}"
export TASK1_UDA_PRED_OUT="${PRED_OUT}"
export TASK1_UDA_CASES_JSON="${PRED_CASES}"
export TASK1_UDA_NNUNET_RESULTS="${NN_RESULTS}"
export TASK1_CUDA_VISIBLE_DEVICES="${TASK1_CUDA_VISIBLE_DEVICES:-0,1,3}"
export TASK1_UDA_PRED_PER_GPU="${TASK1_UDA_PRED_PER_GPU:-3}"

echo "[b1-eval-9fold] predict …"
bash "${CTRL}/ICLR2026/scripts/psma_uda_predict_train.sh"

GT_DIR="${WORK}/nnUNet_raw/Dataset228_AutoPETIV_Task1_2ch/labelsTr"
echo "[b1-eval-9fold] score dice …"
DETAIL_JSON="${OUT}/score_detail.json"
python3 "${CTRL}/ICLR2026/scripts/score_pred_dice_vs_gt.py" \
  --cases-json "${PRED_CASES}" \
  --pred-dir "${PRED_OUT}/pred" \
  --gt-dir "${GT_DIR}" \
  --out-json "${DETAIL_JSON}" \
  --tag "baseline1_fdg_only_psma_val_9fold" \
  --workers 12

# 9fold-shaped aggregate (same dice for every fold — shared val, frozen model)
export FOLDS_CSV DETAIL_JSON SCORE_JSON SCORE_JSON_ALIAS VIS_JSON B1_STAMP STAMP CKPT
python3 - <<'PY'
import json
import os
from pathlib import Path

score = json.loads(Path(os.environ["DETAIL_JSON"]).read_text(encoding="utf-8"))
md = score["mean_dice"]
folds = [int(x) for x in os.environ["FOLDS_CSV"].split(",") if x.strip() != ""]
agg = {
    "protocol": "baseline1_fdg_only_zero_shot_psma_val_9fold",
    "baseline1_stamp": os.environ["B1_STAMP"],
    "eval_stamp": os.environ["STAMP"],
    "ckpt": os.environ["CKPT"],
    "n_val": score["n_scored"],
    "mean_dice": md,
    "mean_dice_positive": score.get("mean_dice_positive"),
    "folds": {str(f): {"best_val_dice": md, "note": "shared_psma_val"} for f in folds},
    "fold_mean": md,
    "n_folds": len(folds),
}
text = json.dumps(agg, indent=2) + "\n"
Path(os.environ["SCORE_JSON"]).write_text(text, encoding="utf-8")
Path(os.environ["SCORE_JSON_ALIAS"]).write_text(text, encoding="utf-8")
Path(os.environ["VIS_JSON"]).write_text(text, encoding="utf-8")
print(text)
PY

echo "STAMP=${STAMP}" > "${LOG_DIR}/iclr2026_baseline1_fdg_eval_psma_9fold_${STAMP}.txt"
echo "[b1-eval-9fold] DONE → ${SCORE_JSON}"
echo "[b1-eval-9fold] VIS → ${VIS_JSON}"
