#!/usr/bin/env bash
# nnUNet PSMA fc70 TEST20 — single run (fold 0), one Dice score.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
ICLR_VIS="${ROOT}/ICLR2026/vis"

PARENT="${PARENT_STAMP:-${TASK1_NNUNET_RESULTS_STAMP_NAME:-}}"
[[ -n "${PARENT}" ]] || { echo "[error] set PARENT_STAMP" >&2; exit 1; }
FOLD="${TASK1_PSMA_FC70_FOLD:-0}"
GPU="${TASK1_PSMA_FC70_GPU:-0}"
PRED_PER_GPU="${TASK1_UDA_PRED_PER_GPU:-5}"

DS="Dataset228_AutoPETIV_Task1_2ch"
TF="nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres"
STAMP="${PARENT}_f${FOLD}"
FOLD_DIR="${WORK}/nnUNet_results/${STAMP}/${DS}/${TF}/fold_0"
CASES_JSON="${ROOT}/ICLR2026/data/splits_mae_psma_test20.json"
GT_DIR="${WORK}/nnUNet_raw/${DS}/labelsTr"
EVAL_ROOT="${WORK}/nnUNet_results/${PARENT}/psma_test20_eval"
mkdir -p "${EVAL_ROOT}"

ckpt="$(python3 - <<PY
from pathlib import Path
fd = Path("${FOLD_DIR}")
for c in ("checkpoint_best.pth", "checkpoint_final.pth"):
    p = fd / c
    if p.is_file():
        print(p)
        break
PY
)"
[[ -n "${ckpt}" && -f "${ckpt}" ]] || { echo "[error] no ckpt ${FOLD_DIR}" >&2; exit 1; }

pred_out="${EVAL_ROOT}/fold${FOLD}"
detail="${pred_out}/score_detail.json"
mkdir -p "${pred_out}"
pred_cases="${pred_out}/cases_test_as_train.json"

python3 - <<PY
import json
from pathlib import Path
raw = json.loads(Path("${CASES_JSON}").read_text())
cases = list(raw[0].get("test") or raw[0].get("val") or raw[0].get("train") or []) if isinstance(raw, list) else list(raw.get("test") or [])
Path("${pred_cases}").write_text(json.dumps([{"train": cases, "val": []}], indent=2) + "\n")
Path("${pred_out}/ckpt_used.json").write_text(json.dumps({"ckpt": Path("${ckpt}").name, "ckpt_path": "${ckpt}"}, indent=2) + "\n")
print(f"[test20-fc70] n={len(cases)} ckpt={Path('${ckpt}').name}")
PY

export TASK1_UDA_CKPT="${ckpt}"
export TASK1_UDA_PRED_OUT="${pred_out}/predict"
export TASK1_UDA_CASES_JSON="${pred_cases}"
export TASK1_UDA_NNUNET_RESULTS="${WORK}/nnUNet_results/${STAMP}"
export TASK1_CUDA_VISIBLE_DEVICES="${GPU}"
export TASK1_UDA_PRED_PER_GPU="${PRED_PER_GPU}"
bash "${ROOT}/ICLR2026/scripts/psma_uda_predict_train.sh"
docker run --rm -v "${ROOT}:${ROOT}" -v "${DATA}:${DATA}" iclr2026_3dmae_petct:cu118 \
  python3 "${ROOT}/ICLR2026/scripts/score_pred_dice_vs_gt.py" \
    --cases-json "${pred_cases}" --pred-dir "${pred_out}/predict/pred" --gt-dir "${GT_DIR}" \
    --out-json "${detail}" --tag "nnunet_psma_fc70_test20" --workers 8

AGG="${ICLR_VIS}/aggregate_nnunet_psma_fc70_${PARENT}.json"
python3 - <<PY
import json
from pathlib import Path
d = json.loads(Path("${detail}").read_text())
md = float(d["mean_dice"])
agg = {
    "protocol": "fc70_nnunet_psma_from_fdg",
    "eval_split": "PSMA_TEST20",
    "single_run": True,
    "parent_stamp": "${PARENT}",
    "folds": {"0": {"test_dice": md, "n_test": d.get("n_scored")}},
    "fold_mean": md,
    "fold_std": 0.0,
    "mean": md,
}
Path("${AGG}").write_text(json.dumps(agg, indent=2) + "\n")
Path("${WORK}/nnUNet_results/${PARENT}/aggregate_test20_dice_fc70.json").write_text(json.dumps(agg, indent=2) + "\n")
print(json.dumps(agg, indent=2))
PY
echo "[test20-fc70] DONE ${AGG}"
