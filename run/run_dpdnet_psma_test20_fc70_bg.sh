#!/usr/bin/env bash
# DpDNet PSMA fc70 TEST20 — single run (fold 0), Dataset240 STUNet ckpt.
#   PARENT_STAMP=<parent> TASK1_PSMA_FC70_GPU=0 bash ICLR2026/run/run_dpdnet_psma_test20_fc70_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
ICLR_VIS="${CTRL}/ICLR2026/vis"
DPD="${CTRL}/ICLR2026/third_party/DpDNet"
IMAGE="${TASK1_NNUNET_IMAGE:-autopet_baseline:latest}"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
mkdir -p "${ICLR_VIS}"

DATASET_ID="${DATASET_ID:-240}"
DS="Dataset${DATASET_ID}_DpDNet_PSMA_2ch"
SRC_DS="Dataset228_AutoPETIV_Task1_2ch"
TRAINER="${TRAINER:-STUNetTrainer_small_prompt}"
CONFIG="${CONFIG:-3d_fullres}"
TF="${TRAINER}__nnUNetPlans__${CONFIG}"
FOLD="${TASK1_PSMA_FC70_FOLD:-0}"
GPU="${TASK1_PSMA_FC70_GPU:-0}"
PROMPT_PREFIX="${TASK1_DPDNET_PROMPT_PREFIX:-lymp}"

PARENT="${PARENT_STAMP:-${TASK1_NNUNET_RESULTS_STAMP_NAME:-}}"
[[ -n "${PARENT}" ]] || { echo "[error] set PARENT_STAMP" >&2; exit 1; }
PARENT="${PARENT%_f${FOLD}}"

STAMP="${PARENT}_f${FOLD}"
FOLD_DIR="${WORK}/nnUNet_results/${STAMP}/${DS}/${TF}/fold_${FOLD}"
# some runs keep weights in fold_0 even when stamp already has _f0
[[ -d "${FOLD_DIR}" ]] || FOLD_DIR="${WORK}/nnUNet_results/${STAMP}/${DS}/${TF}/fold_0"

ckpt=""
for c in checkpoint_best.pth checkpoint_final.pth checkpoint_latest.pth; do
  if [[ -f "${FOLD_DIR}/${c}" ]]; then
    ckpt="${c}"
    break
  fi
done
[[ -n "${ckpt}" ]] || { echo "[error] no Dataset240 ckpt under ${FOLD_DIR}" >&2; exit 1; }

CASES_JSON="${ROOT}/ICLR2026/data/splits_mae_psma_test20.json"
GT_DIR="${WORK}/nnUNet_raw/${SRC_DS}/labelsTr"
RAW_IMG="${WORK}/nnUNet_raw/${SRC_DS}/imagesTr"
EVAL_ROOT="${WORK}/nnUNet_results/${PARENT}/psma_test20_eval"
pred_out="${EVAL_ROOT}/fold${FOLD}"
detail="${pred_out}/score_detail.json"
log="${ICLR_VIS}/nohup_dpdnet_test20_fc70_f${FOLD}_${PARENT}.log"
tmp_in="${pred_out}/imagesTs_lymp"
tmp_pred="${pred_out}/predict_lymp"
pred_flat="${pred_out}/predict"
mkdir -p "${pred_out}"
chmod -R a+rwX "${pred_out}" || true

echo "[dpdnet-fc70-test20] PARENT=${PARENT} stamp=${STAMP} fold=${FOLD} gpu=${GPU} ckpt=${ckpt}" | tee "${log}"

rm -rf "${tmp_in}" "${tmp_pred}" "${pred_flat}"
mkdir -p "${tmp_in}" "${tmp_pred}" "${pred_flat}"

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
print(f"[dpdnet-fc70-test20] n={n} ckpt=${ckpt}")
Path("${pred_out}/cases.json").write_text(json.dumps({"cases": cases, "prompt_prefix": pref}, indent=2) + "\n")
Path("${pred_out}/cases_test_as_train.json").write_text(json.dumps([{"train": cases, "val": []}], indent=2) + "\n")
Path("${pred_out}/ckpt_used.json").write_text(json.dumps({"ckpt": "${ckpt}", "fold_dir": "${FOLD_DIR}"}, indent=2) + "\n")
PY

docker run --rm --user root \
  --gpus "\"device=${GPU}\"" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HOME=/home/algorithm \
  -e nnUNet_raw="${WORK}/nnUNet_raw" \
  -e nnUNet_preprocessed="${WORK}/nnUNet_preprocessed" \
  -e nnUNet_results="${WORK}/nnUNet_results/${STAMP}" \
  -e PYTHONPATH="${DPD}:/home/algorithm/.local/lib/python3.11/site-packages" \
  -v "${CTRL}:${CTRL}" \
  -v "${DATA}:${DATA}" \
  --shm-size=16g \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "mkdir -p '${tmp_pred}' && chmod -R a+rwX '${pred_out}' && python '${CTRL}/ICLR2026/scripts/dpdnet_predict_prompt_cli.py' -i '${tmp_in}' -o '${tmp_pred}' -d ${DATASET_ID} -c ${CONFIG} -tr ${TRAINER} -f ${FOLD} -chk ${ckpt} -npp 2 -nps 2 --disable_tta" \
  >>"${log}" 2>&1

python3 - <<PY
from pathlib import Path
import shutil
pref = "${PROMPT_PREFIX}_"
src = Path("${tmp_pred}")
dst = Path("${pred_flat}")
dst.mkdir(parents=True, exist_ok=True)
n = 0
for p in src.glob("*.nii.gz"):
    name = p.name
    if name.startswith(pref):
        name = name[len(pref):]
    shutil.copy2(p, dst / name)
    n += 1
print(f"[dpdnet-fc70-test20] flattened n={n}")
PY

docker run --rm \
  -v "${ROOT}:${ROOT}" -v "${DATA}:${DATA}" \
  iclr2026_3dmae_petct:cu118 \
  python3 "${ROOT}/ICLR2026/scripts/score_pred_dice_vs_gt.py" \
    --cases-json "${pred_out}/cases_test_as_train.json" \
    --pred-dir "${pred_flat}" \
    --gt-dir "${GT_DIR}" \
    --out-json "${detail}" \
    --tag "dpdnet_psma_fc70_test20" \
    --workers 8 \
  >>"${log}" 2>&1

AGG="${ICLR_VIS}/aggregate_dpdnet_psma_fc70_${PARENT}.json"
python3 - <<PY
import json
from pathlib import Path
d = json.loads(Path("${detail}").read_text())
md = float(d["mean_dice"])
fp, fn = d.get("fp_rate", d.get("mean_fp")), d.get("fn_rate", d.get("mean_fn"))
agg = {
    "protocol": "fc70_dpdnet_psma_from_fdg",
    "eval_split": "PSMA_TEST20",
    "single_run": True,
    "parent_stamp": "${PARENT}",
    "folds": {"0": {"test_dice": md, "n_test": d.get("n_scored")}},
    "fold_mean": md,
    "fold_std": 0.0,
    "mean": md,
    "mean_fp": fp,
    "mean_fn": fn,
    "ckpt": "${ckpt}",
}
Path("${AGG}").write_text(json.dumps(agg, indent=2) + "\n")
Path("${WORK}/nnUNet_results/${PARENT}/aggregate_test20_dice_fc70.json").write_text(json.dumps(agg, indent=2) + "\n")
print(json.dumps(agg, indent=2))
PY

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"dpdnet\":{\"psma_fc70\":{\"status\":\"done\",\"stamp\":\"${PARENT}\",\"note\":\"TEST20 DONE · fc70 single\"}}},\"updated_note\":\"DpDNet fc70 TEST20 done\"}" || true

echo "[dpdnet-fc70-test20] DONE ${AGG}"
