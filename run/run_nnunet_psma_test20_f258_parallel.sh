#!/usr/bin/env bash
# nnUNet PSMA TEST20: 各站(折)一卡并行 · 每卡内分片 TASK1_UDA_PRED_PER_GPU（默认 5）
#
# Required:
#   PARENT_STAMP / TASK1_NNUNET_RESULTS_STAMP_NAME  e.g. ..._psma_fs50_f258_..._gpu013
# Optional:
#   TASK1_FOLDS=2,5,8
#   TASK1_FOLD_GPUS=2:0,5:1,8:3
#   TASK1_UDA_PRED_PER_GPU=5
#   TASK1_TEST_SKIP_DONE=1   skip folds with score_detail.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK1_REPO_ROOT="${ROOT}"
export TASK1_BASE="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
export WORK_DIR="${WORK_DIR:-${TASK1_BASE}/task1_train_workspace}"
WORK="${WORK_DIR}"
ICLR_VIS="${TASK1_LOSS_OUT_DIR:-${ROOT}/ICLR2026/vis}"
mkdir -p "${ICLR_VIS}"

DATASET_ID="${DATASET_ID:-228}"
DS="Dataset${DATASET_ID}_AutoPETIV_Task1_2ch"
TRAINER="${TRAINER:-nnUNetTrainer_Task1StdTrainVal50}"
PLANS_ID="${PLANS_ID:-nnUNetPlans}"
CONFIG="${CONFIG:-3d_fullres}"
TF="${TRAINER}__${PLANS_ID}__${CONFIG}"
FOLDS_CSV="${TASK1_FOLDS:-2,5,8}"
FOLD_GPUS_CSV="${TASK1_FOLD_GPUS:-2:0,5:1,8:3}"
PRED_PER_GPU="${TASK1_UDA_PRED_PER_GPU:-5}"
SKIP_DONE="${TASK1_TEST_SKIP_DONE:-1}"

PARENT="${PARENT_STAMP:-${TASK1_NNUNET_RESULTS_STAMP_NAME:-}}"
[[ -n "${PARENT}" ]] || { echo "[error] set PARENT_STAMP" >&2; exit 1; }

IFS=',' read -r -a FOLDS <<< "${FOLDS_CSV}"
declare -A GPU_OF
IFS=',' read -r -a _pairs <<< "${FOLD_GPUS_CSV}"
for p in "${_pairs[@]}"; do
  GPU_OF["${p%%:*}"]="${p##*:}"
done

CASES_JSON="${ROOT}/ICLR2026/data/splits_mae_psma_test20.json"
GT_DIR="${WORK}/nnUNet_raw/${DS}/labelsTr"
AGG_ROOT="${WORK}/nnUNet_results/${PARENT}"
EVAL_ROOT="${AGG_ROOT}/psma_test20_eval"
mkdir -p "${EVAL_ROOT}" "${ICLR_VIS}"

echo "[nnunet-test20] PARENT=${PARENT} folds=${FOLDS_CSV} map=${FOLD_GPUS_CSV} per_gpu=${PRED_PER_GPU}"

_run_one_fold() {
  local fold="$1" gpu="$2"
  local stamp="${PARENT}_f${fold}"
  local fold_dir="${WORK}/nnUNet_results/${stamp}/${DS}/${TF}/fold_0"
  local ckpt=""
  ckpt="$(python3 - <<PY
from pathlib import Path
fd = Path("${fold_dir}")
best, final = fd / "checkpoint_best.pth", fd / "checkpoint_final.pth"
nan_best = False
real_best = False
for log in sorted(fd.glob("training_log*.txt")):
    for line in log.read_text(errors="ignore").splitlines():
        if "New best EMA" not in line:
            continue
        low = line.lower()
        if "nan" in low:
            nan_best = True
        else:
            real_best = True
if best.is_file() and (real_best or not nan_best):
    print(best)
elif final.is_file():
    print(final)
elif best.is_file():
    print(best)
PY
)"
  [[ -n "${ckpt}" && -f "${ckpt}" ]] || { echo "[error] no ckpt fold${fold}" >&2; return 1; }

  local pred_out="${EVAL_ROOT}/fold${fold}"
  local pred_cases="${pred_out}/cases_test_as_train.json"
  local detail="${pred_out}/score_detail.json"
  local log="${ICLR_VIS}/nohup_nnunet_test20_f${fold}_${PARENT}.log"
  mkdir -p "${pred_out}"

  if [[ "${SKIP_DONE}" == "1" && -f "${detail}" ]]; then
    echo "[nnunet-test20] skip fold${fold} (score exists)" | tee -a "${log}"
    return 0
  fi

  # Another job already predicting this fold (e.g. GPU fill-in) — wait, do not double-write.
  # Match real predictors only (python/docker nnUNetv2_predict), not pgrep/bash wrappers.
  _fold_predict_live() {
    pgrep -af "nnUNetv2_predict" 2>/dev/null | grep -F "psma_test20_eval/fold${fold}/" | grep -vE 'pgrep|grep|bash -lc|timeout ' >/dev/null
  }
  if _fold_predict_live; then
    echo "[nnunet-test20] fold${fold} already predicting — wait for score_detail" | tee -a "${log}"
    while [[ ! -f "${detail}" ]]; do
      _fold_predict_live || break
      sleep 30
    done
    if [[ -f "${detail}" ]]; then
      echo "[nnunet-test20] fold${fold} done by peer" | tee -a "${log}"
      return 0
    fi
    echo "[nnunet-test20] fold${fold} peer exited without score — resume locally" | tee -a "${log}"
  fi

  # resume: keep partial shard/flat preds; uda-predict skips complete nii+npz
  mkdir -p "${pred_out}"

  python3 - <<PY
import json
from pathlib import Path
raw = json.loads(Path("${CASES_JSON}").read_text())
if isinstance(raw, list) and raw and isinstance(raw[0], dict):
    cases = list(raw[0].get("test") or raw[0].get("val") or raw[0].get("train") or [])
elif isinstance(raw, dict):
    cases = list(raw.get("test") or raw.get("val") or raw.get("train") or raw.get("cases") or [])
else:
    cases = list(raw)
Path("${pred_cases}").write_text(json.dumps([{"train": cases, "val": []}], indent=2) + "\n")
ckpt_p = Path("${ckpt}")
Path("${pred_out}/ckpt_used.json").write_text(
    json.dumps({"ckpt": ckpt_p.name, "ckpt_path": str(ckpt_p)}, indent=2) + "\n"
)
print(f"[eval-TEST20] fold${fold} gpu=${gpu} per_gpu=${PRED_PER_GPU} n={len(cases)} ckpt=${ckpt}")
PY

  python3 - <<PY
import json
from pathlib import Path
import sys
sys.path.insert(0, "${ROOT}/ICLR2026/scripts")
from test20_ckpt_ep import nnunet_fold_ckpt_ep
ckpt_p = Path("${ckpt}")
parent = "${PARENT}"
ep = nnunet_fold_ckpt_ep(parent, int("${fold}"), Path("${WORK}"), ckpt_p.name)
side = Path("${pred_out}/ckpt_used.json")
data = json.loads(side.read_text()) if side.is_file() else {}
if ep is not None:
    data["ckpt_ep"] = ep
side.write_text(json.dumps(data, indent=2) + "\n")
PY

  echo "[nnunet-test20] fold${fold} → GPU${gpu} × ${PRED_PER_GPU} shards" | tee -a "${log}"
  (
    export TASK1_UDA_CKPT="${ckpt}"
    export TASK1_UDA_PRED_OUT="${pred_out}/predict"
    export TASK1_UDA_CASES_JSON="${pred_cases}"
    export TASK1_UDA_NNUNET_RESULTS="${WORK}/nnUNet_results/${stamp}"
    export TASK1_CUDA_VISIBLE_DEVICES="${gpu}"
    export TASK1_UDA_PRED_PER_GPU="${PRED_PER_GPU}"
    bash "${ROOT}/ICLR2026/scripts/psma_uda_predict_train.sh"
    docker run --rm \
      -v "${ROOT}:${ROOT}" -v "${TASK1_BASE}:${TASK1_BASE}" \
      iclr2026_3dmae_petct:cu118 \
      python3 "${ROOT}/ICLR2026/scripts/score_pred_dice_vs_gt.py" \
        --cases-json "${pred_cases}" \
        --pred-dir "${pred_out}/predict/pred" \
        --gt-dir "${GT_DIR}" \
        --out-json "${detail}" \
        --tag "nnunet_psma_fs50_test20_f${fold}" \
        --workers 8
  ) >>"${log}" 2>&1
}

pids=()
for fold in "${FOLDS[@]}"; do
  gpu="${GPU_OF[${fold}]:-}"
  [[ -n "${gpu}" ]] || { echo "[error] no GPU for fold ${fold}" >&2; exit 1; }
  _run_one_fold "${fold}" "${gpu}" &
  pids+=($!)
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done
[[ "${rc}" -eq 0 ]] || { echo "[error] some TEST folds failed" >&2; exit 1; }

AGG_JSON="${AGG_ROOT}/aggregate_test20_dice_f258.json"
VIS_JSON="${ICLR_VIS}/aggregate_nnunet_psma_fs50_f258_${PARENT}.json"
python3 - <<PY
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, "${ROOT}/ICLR2026/scripts")
from test20_ckpt_ep import nnunet_fold_ckpt_ep

ALL_FOLDS = tuple(range(9))
run_folds = [int(x) for x in "${FOLDS_CSV}".split(",") if x.strip()]
root = Path("${EVAL_ROOT}")
work = Path("${WORK}")
parent = "${PARENT}"
fold_map, vals, fep = {}, [], {}
fp_vals, fn_vals = [], []
sum_fp = sum_fn = sum_neg = sum_pos = 0
for f in ALL_FOLDS:
    det = root / f"fold{f}" / "score_detail.json"
    if not det.is_file():
        continue
    d = json.loads(det.read_text())
    md = float(d["mean_dice"])
    ckpt_name = None
    used = root / f"fold{f}" / "ckpt_used.json"
    if used.is_file():
        try:
            ckpt_name = json.loads(used.read_text()).get("ckpt")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            ckpt_name = None
    ckpt_ep = nnunet_fold_ckpt_ep(parent, f, work, ckpt_name)
    fp = d.get("fp_rate", d.get("mean_fp"))
    fn = d.get("fn_rate", d.get("mean_fn"))
    fold_map[str(f)] = {
        "test_dice": md,
        "best_val_dice": md,
        "mean_dice_positive": d.get("mean_dice_positive"),
        "n_test": d.get("n_scored"),
        "ckpt_stamp": f"{parent}_f{f}",
        "ckpt": ckpt_name,
        "ckpt_ep": ckpt_ep,
        "fp_rate": float(fp) if isinstance(fp, (int, float)) and fp == fp else None,
        "fn_rate": float(fn) if isinstance(fn, (int, float)) and fn == fn else None,
        "mean_fp": float(fp) if isinstance(fp, (int, float)) and fp == fp else None,
        "mean_fn": float(fn) if isinstance(fn, (int, float)) and fn == fn else None,
    }
    if ckpt_ep is not None:
        fep[str(f)] = ckpt_ep
    vals.append(md)
    if isinstance(fp, (int, float)) and fp == fp:
        fp_vals.append(float(fp))
    if isinstance(fn, (int, float)) and fn == fn:
        fn_vals.append(float(fn))
    sum_fp += int(d.get("sum_fp") or 0)
    sum_fn += int(d.get("sum_fn") or 0)
    sum_neg += int(d.get("sum_neg_voxels") or 0)
    sum_pos += int(d.get("sum_pos_voxels") or 0)
if not vals:
    raise SystemExit("no score_detail.json under extra-fold TEST20 eval dir")
mean_fp = (sum_fp / sum_neg) if sum_neg > 0 else (sum(fp_vals) / len(fp_vals) if fp_vals else None)
mean_fn = (sum_fn / sum_pos) if sum_pos > 0 else (sum(fn_vals) / len(fn_vals) if fn_vals else None)
ckpt_files = [fv.get("ckpt") for fv in fold_map.values() if fv.get("ckpt")]
# VIS name keeps fs50_f258 prefix for board ingest compatibility; stamp encodes fewshot.
_stage_tag = "fs50"
if "_fs10_" in parent:
    _stage_tag = "fs10"
elif "_fs5_" in parent:
    _stage_tag = "fs5"
vis_json = Path("${ICLR_VIS}") / f"aggregate_nnunet_psma_{_stage_tag}_f258_{parent}.json"
agg = {
    "protocol": "fewshot50_nnunet_psma_finetune_from_baseline1_fdg",
    "eval_split": "PSMA_TEST20",
    "test_layout": "1gpu_per_fold_parallel",
    "pred_per_gpu": int("${PRED_PER_GPU}"),
    "fold_gpus": "${FOLD_GPUS_CSV}",
    "parent_stamp": parent,
    "folds": fold_map,
    "fold_ckpt_ep": fep,
    "fold_mean": float(sum(vals) / len(vals)) if vals else None,
    "fold_std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
    "ckpt": ckpt_files[0] if ckpt_files else None,
    "fp_rate": mean_fp,
    "fn_rate": mean_fn,
    "mean_fp": mean_fp,
    "mean_fn": mean_fn,
    "sum_fp": sum_fp,
    "sum_fn": sum_fn,
    "sum_neg_voxels": sum_neg,
    "sum_pos_voxels": sum_pos,
}
Path("${AGG_JSON}").write_text(json.dumps(agg, indent=2) + "\n")
Path("${AGG_ROOT}/aggregate_val_dice_f258.json").write_text(json.dumps(agg, indent=2) + "\n")
# also keep legacy fs50-named copy so older ingest paths still find it
Path("${VIS_JSON}").write_text(json.dumps(agg, indent=2) + "\n")
vis_json.write_text(json.dumps(agg, indent=2) + "\n")
print(json.dumps(agg, indent=2))
PY

echo "[nnunet-test20] ALL DONE → ${AGG_JSON}"
