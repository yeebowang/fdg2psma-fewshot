#!/usr/bin/env bash
# DpDNet PSMA TEST20 · folds 2/5/8 · checkpoint_best（max val Pseudo dice）
#
# Required: PARENT_STAMP
# Optional: TASK1_FOLDS / TASK1_FOLD_GPUS / TASK1_UDA_PRED_PER_GPU
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
ICLR_VIS="${CTRL}/ICLR2026/vis"
DPD="${CTRL}/ICLR2026/third_party/DpDNet"
IMAGE="${TASK1_NNUNET_IMAGE:-autopet_baseline:latest}"
mkdir -p "${ICLR_VIS}"

DATASET_ID="${DATASET_ID:-240}"
DS="Dataset${DATASET_ID}_DpDNet_PSMA_2ch"
SRC_DS="Dataset228_AutoPETIV_Task1_2ch"
TRAINER="${TRAINER:-STUNetTrainer_small_prompt}"
CONFIG="${CONFIG:-3d_fullres}"
TF="${TRAINER}__nnUNetPlans__${CONFIG}"
FOLDS_CSV="${TASK1_FOLDS:-2,5,8}"
FOLD_GPUS_CSV="${TASK1_FOLD_GPUS:-2:0,5:1,8:3}"
PRED_PER_GPU="${TASK1_UDA_PRED_PER_GPU:-5}"
SKIP_DONE="${TASK1_TEST_SKIP_DONE:-1}"
PROMPT_PREFIX="${TASK1_DPDNET_PROMPT_PREFIX:-lymp}"

PARENT="${PARENT_STAMP:-${TASK1_NNUNET_RESULTS_STAMP_NAME:-}}"
[[ -n "${PARENT}" ]] || { echo "[error] set PARENT_STAMP" >&2; exit 1; }

IFS=',' read -r -a FOLDS <<< "${FOLDS_CSV}"
declare -A GPU_OF
IFS=',' read -r -a _pairs <<< "${FOLD_GPUS_CSV}"
for p in "${_pairs[@]}"; do
  GPU_OF["${p%%:*}"]="${p##*:}"
done

CASES_JSON="${ROOT}/ICLR2026/data/splits_mae_psma_test20.json"
GT_DIR="${WORK}/nnUNet_raw/${SRC_DS}/labelsTr"
RAW_IMG="${WORK}/nnUNet_raw/${SRC_DS}/imagesTr"
AGG_ROOT="${WORK}/nnUNet_results/${PARENT}"
EVAL_ROOT="${AGG_ROOT}/psma_test20_eval"
mkdir -p "${EVAL_ROOT}"

echo "[dpdnet-test20] PARENT=${PARENT} folds=${FOLDS_CSV} map=${FOLD_GPUS_CSV}"

_run_one_fold() {
  local fold="$1" gpu="$2"
  local stamp="${PARENT}_f${fold}"
  local fold_dir="${WORK}/nnUNet_results/${stamp}/${DS}/${TF}/fold_${fold}"
  local model_dir="${WORK}/nnUNet_results/${stamp}/${DS}/${TF}"
  local ckpt=""
  for c in checkpoint_best.pth checkpoint_final.pth; do
    [[ -f "${fold_dir}/${c}" ]] && { ckpt="${c}"; break; }
  done
  [[ -n "${ckpt}" ]] || { echo "[error] no ckpt fold${fold} under ${fold_dir}" >&2; return 1; }

  local pred_out="${EVAL_ROOT}/fold${fold}"
  local detail="${pred_out}/score_detail.json"
  local log="${ICLR_VIS}/nohup_dpdnet_test20_f${fold}_${PARENT}.log"
  local tmp_in="${pred_out}/imagesTs_lymp"
  local tmp_pred="${pred_out}/predict_lymp"
  local pred_flat="${pred_out}/predict"
  mkdir -p "${pred_out}"

  if [[ "${SKIP_DONE}" == "1" && -f "${detail}" ]]; then
    echo "[dpdnet-test20] skip fold${fold} (score exists)" | tee -a "${log}"
    return 0
  fi

  rm -rf "${tmp_in}" "${tmp_pred}" "${pred_flat}"
  mkdir -p "${tmp_in}" "${tmp_pred}" "${pred_flat}"
  # docker (algorithm/non-root) must write predict sidecars under these dirs
  chmod -R a+rwX "${pred_out}" || true

  python3 - <<PY
import json, os
from pathlib import Path
raw = json.loads(Path("${CASES_JSON}").read_text())
if isinstance(raw, list) and raw and isinstance(raw[0], dict):
    cases = list(raw[0].get("test") or raw[0].get("val") or raw[0].get("train") or [])
elif isinstance(raw, dict):
    cases = list(raw.get("test") or raw.get("val") or raw.get("train") or raw.get("cases") or [])
else:
    cases = list(raw)
pref = "${PROMPT_PREFIX}"
src = Path("${RAW_IMG}")
dst = Path("${tmp_in}")
n = 0
for case in cases:
    for ch in (0, 1):
        s = src / f"{case}_{ch:04d}.nii.gz"
        d = dst / f"{pref}_{case}_{ch:04d}.nii.gz"
        if not s.is_file():
            raise SystemExit(f"missing {s}")
        if d.exists() or d.is_symlink():
            d.unlink()
        os.symlink(s, d)
    n += 1
print(f"[dpdnet-test20] fold${fold} gpu=${gpu} n={n} ckpt=${ckpt} (max val Dice best)")
Path("${pred_out}/cases.json").write_text(json.dumps({"cases": cases, "prompt_prefix": pref}, indent=2) + "\n")
Path("${pred_out}/cases_test_as_train.json").write_text(json.dumps([{"train": cases, "val": []}], indent=2) + "\n")
PY

  echo "[dpdnet-test20] fold${fold} → GPU${gpu} predict (${ckpt})" | tee -a "${log}"
  # STUNetTrainer_small_prompt needs prompt predictor (not stock nnUNetv2_predict).
  # --user root: host-created eval dirs may not be writable by image default user.
  docker run --rm --user root \
    --gpus "\"device=${gpu}\"" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e HOME=/home/algorithm \
    -e nnUNet_raw="${WORK}/nnUNet_raw" \
    -e nnUNet_preprocessed="${WORK}/nnUNet_preprocessed" \
    -e nnUNet_results="${WORK}/nnUNet_results/${stamp}" \
    -e PYTHONPATH="${DPD}:/home/algorithm/.local/lib/python3.11/site-packages" \
    -v "${CTRL}:${CTRL}" \
    -v "${DATA}:${DATA}" \
    --shm-size=16g \
    --entrypoint bash \
    "${IMAGE}" \
    -lc "mkdir -p '${tmp_pred}' && chmod -R a+rwX '${pred_out}' && python '${CTRL}/ICLR2026/scripts/dpdnet_predict_prompt_cli.py' -i '${tmp_in}' -o '${tmp_pred}' -d ${DATASET_ID} -c ${CONFIG} -tr ${TRAINER} -f ${fold} -chk ${ckpt} -npp 2 -nps 2 --disable_tta" \
    >>"${log}" 2>&1

  # strip prompt prefix from prediction filenames for GT matching
  python3 - <<PY
from pathlib import Path
import shutil
pref = "${PROMPT_PREFIX}_"
src = Path("${tmp_pred}")
dst = Path("${pred_flat}")
dst.mkdir(parents=True, exist_ok=True)
for p in src.glob("*.nii.gz"):
    name = p.name
    if name.startswith(pref):
        name = name[len(pref):]
    shutil.copy2(p, dst / name)
print(f"[dpdnet-test20] flattened preds → {dst} n={len(list(dst.glob('*.nii.gz')))}")
PY

  docker run --rm \
    -v "${ROOT}:${ROOT}" -v "${DATA}:${DATA}" \
    iclr2026_3dmae_petct:cu118 \
    python3 "${ROOT}/ICLR2026/scripts/score_pred_dice_vs_gt.py" \
      --cases-json "${pred_out}/cases_test_as_train.json" \
      --pred-dir "${pred_flat}" \
      --gt-dir "${GT_DIR}" \
      --out-json "${detail}" \
      --tag "dpdnet_psma_fs50_test20_f${fold}" \
      --workers 8 \
    >>"${log}" 2>&1

  echo "[dpdnet-test20] fold${fold} done detail=${detail}" | tee -a "${log}"
}

pids=()
for fold in "${FOLDS[@]}"; do
  gpu="${GPU_OF[${fold}]:-}"
  [[ -n "${gpu}" ]] || { echo "[error] no GPU for fold ${fold}" >&2; exit 1; }
  _run_one_fold "${fold}" "${gpu}" &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

AGG_JSON="${AGG_ROOT}/aggregate_test20_dice_f258.json"
python3 - <<PY
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, "${ROOT}/ICLR2026/scripts")
from test20_ckpt_ep import dpdnet_fold_ckpt_ep

ALL_FOLDS = tuple(range(9))
root = Path("${EVAL_ROOT}")
work = Path("${WORK}")
parent = "${PARENT}"
fold_map, vals, fep = {}, [], {}
for f in ALL_FOLDS:
    det = root / f"fold{f}" / "score_detail.json"
    if not det.is_file():
        continue
    d = json.loads(det.read_text())
    md = float(d["mean_dice"])
    ckpt_ep = dpdnet_fold_ckpt_ep(parent, f, work, "checkpoint_best.pth")
    fold_map[str(f)] = {
        "test_dice": md,
        "mean_dice_positive": d.get("mean_dice_positive"),
        "n_test": d.get("n_scored"),
        "ckpt_stamp": f"{parent}_f{f}",
        "ckpt_policy": "checkpoint_best(max val Pseudo dice)",
        "ckpt_ep": ckpt_ep,
    }
    if ckpt_ep is not None:
        fep[str(f)] = ckpt_ep
    vals.append(md)
if not vals:
    raise SystemExit("no score_detail.json under extra-fold TEST20 eval dir")
agg = {
    "protocol": "fewshot50_dpdnet_psma_finetune_from_fdg",
    "eval_split": "PSMA_TEST20",
    "parent_stamp": parent,
    "folds": fold_map,
    "fold_ckpt_ep": fep,
    "fold_mean": float(sum(vals) / len(vals)) if vals else None,
    "fold_std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
    "ckpt_policy": "checkpoint_best = max val Pseudo dice (val25 every20)",
    "ckpt": "checkpoint_best.pth",
}
Path("${AGG_JSON}").write_text(json.dumps(agg, indent=2) + "\n")
Path("${ICLR_VIS}/aggregate_dpdnet_psma_test20_f258_${PARENT}.json").write_text(json.dumps(agg, indent=2) + "\n")
Path("${ICLR_VIS}/aggregate_dpdnet_psma_fs50_f258_${PARENT}.json").write_text(json.dumps(agg, indent=2) + "\n")
print(json.dumps(agg, indent=2))
PY

echo "[dpdnet-test20] aggregate → ${AGG_JSON}"
