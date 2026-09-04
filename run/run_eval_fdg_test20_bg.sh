#!/usr/bin/env bash
# FDG TEST: shared FDG ckpt → FDG 20% TEST (202 cases) · stratified 70/10/20 test split.
#
#   bash ICLR2026/run/run_eval_fdg_test20_bg.sh
#   METHOD=nnunet|mae|monai|dpdnet|seganypet|proto|all  (default all)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
AGG_DIR="${VIS}/fdg_test20"
EVAL_ROOT="${WORK}/fdg_test20_eval"
TEST_JSON="${CTRL}/ICLR2026/data/splits_fdg_test20.json"
GT_DIR="${TASK1_MAE_LABELS_TR:-${DATA}/dataset1/labelsTr}"
RAW_IMG="${WORK}/nnUNet_raw/Dataset228_AutoPETIV_Task1_2ch/imagesTr"
IMAGE_MAE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"
IMAGE_NN="${TASK1_NNUNET_IMAGE:-autopet_baseline:latest}"
DPD="${CTRL}/ICLR2026/third_party/DpDNet"
SEG_CODE="${CTRL}/ICLR2026/third_party/SegAnyPET/code"
SEG_PIP="${CTRL}/ICLR2026/third_party/seganypet_pip"
CACHE="${TASK1_MAE_FDG_TEST_CACHE:-${WORK}/mae_cache/fdg_test20}"
RAW_MAE_IMG="${TASK1_MAE_IMAGES_TR:-${DATA}/dataset1/imagesTr}"
RAW_MAE_LAB="${TASK1_MAE_LABELS_TR:-${DATA}/dataset1/labelsTr}"
SEG_RAW_ROOT="${TASK1_SEGANY_FDG_RAW:-${WORK}/nnUNet_raw/Dataset221_AutoPETIV_Task1_4ch}"
SEG_TEST_ROOT="${TASK1_SEGANY_FDG_TEST20_ROOT:-${WORK}/seganypet_fdg_test20}"
GPU="${TASK1_CUDA_VISIBLE_DEVICES:-0}"
METHOD="${METHOD:-all}"
SKIP_DONE="${TASK1_TEST_SKIP_DONE:-1}"
STAMP_TZ="${STAMP_TZ:-Asia/Shanghai}"
EVAL_STAMP="${TASK1_FDG_TEST20_STAMP:-$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_fdg_test20_gpu013}"

mkdir -p "${AGG_DIR}" "${EVAL_ROOT}" "${VIS}"

PIPE_LOG="${VIS}/nohup_fdg_test20_${EVAL_STAMP}.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[fdg-test20] EVAL_STAMP=${EVAL_STAMP} METHOD=${METHOD} gpu=${GPU}"

_board_ckpt() {
  local mkey="$1" field="${2:-best_ckpt}"
  python3 - <<PY
import json
from pathlib import Path
board = Path("${BOARD}")
if not board.is_file():
    raise SystemExit(0)
b = json.loads(board.read_text())
fdg = (b.get("methods") or {}).get("${mkey}", {}).get("fdg_pretrain") or {}
v = fdg.get("${field}") or fdg.get("best_ckpt") or ""
print(v)
PY
}

_write_agg() {
  local mkey="$1" json_path="$2"
  python3 - <<PY
import json
from pathlib import Path
d = json.loads(Path("${json_path}").read_text())
d["eval_stamp"] = "${EVAL_STAMP}"
d["protocol"] = "fdg_shared_ckpt_fdg_test20"
d["split"] = "FDG_TEST20"
out = Path("${AGG_DIR}") / "aggregate_${mkey}.json"
out.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
print(f"[fdg-test20] aggregate → {out}")
PY
}

_patch_board_running() {
  local mkey="$1"
  local gpus="${CUDA_VISIBLE_DEVICES:-${TASK1_GPUS:-0,1,3}}"
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --patch-json "{\"updated_note\":\"FDG TEST · RUNNING (GPU ${gpus}) · ${mkey}\",\"methods\":{\"${mkey}\":{\"fdg_test20\":{\"status\":\"running\",\"stamp\":\"${EVAL_STAMP}\",\"device\":\"gpu\",\"gpu_ids\":\"${gpus}\"}}}}" \
    || true
}

_patch_board_done() {
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true
}

_cases_json_train_only() {
  local out="$1"
  python3 - <<PY
import json
from pathlib import Path
raw = json.loads(Path("${TEST_JSON}").read_text())
if isinstance(raw, dict):
    cases = list(raw.get("cases") or raw.get("test") or [])
elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
    cases = list(raw[0].get("test") or raw[0].get("train") or [])
else:
    cases = list(raw)
Path("${out}").write_text(json.dumps([{"train": cases, "val": []}], indent=2) + "\n", encoding="utf-8")
print(f"[fdg-test20] n_test={len(cases)} → ${out}")
PY
}

_eval_nnunet() {
  local mkey="${1:-nnunet}"
  local agg="${AGG_DIR}/aggregate_${mkey}.json"
  [[ "${SKIP_DONE}" == "1" && -f "${agg}" ]] && { echo "[fdg-test20] skip ${mkey} (aggregate exists)"; return 0; }
  _patch_board_running "${mkey}"
  local ckpt
  ckpt="$(_board_ckpt "${mkey}")"
  [[ -z "${ckpt}" ]] && ckpt="${TASK1_NNUNET_FDG_CKPT:-/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth}"
  [[ -f "${ckpt}" ]] || { echo "[error] ${mkey} fdg ckpt missing: ${ckpt}" >&2; return 1; }
  local out="${EVAL_ROOT}/${mkey}"
  local pred="${out}/predict"
  local cases="${out}/cases_test_as_train.json"
  local detail="${out}/score_detail.json"
  mkdir -p "${out}" "${pred}/pred"
  _cases_json_train_only "${cases}"
  export TASK1_BASE="${DATA}" TASK1_UDA_CKPT="${ckpt}" TASK1_UDA_PRED_OUT="${pred}"
  export TASK1_UDA_CASES_JSON="${cases}"
  export TASK1_UDA_NNUNET_RESULTS="$(dirname "$(dirname "$(dirname "$(dirname "${ckpt}")")")")"
  export TASK1_CUDA_VISIBLE_DEVICES="${TASK1_CUDA_VISIBLE_DEVICES:-0,1,3}"
  export TASK1_UDA_PRED_PER_GPU="${TASK1_UDA_PRED_PER_GPU:-3}"
  echo "[fdg-test20] ${mkey} ckpt=${ckpt}"
  bash "${CTRL}/ICLR2026/scripts/psma_uda_predict_train.sh"
  docker run --rm -v "${ROOT}:${ROOT}" -v "${DATA}:${DATA}" "${IMAGE_MAE}" \
    python3 "${ROOT}/ICLR2026/scripts/score_pred_dice_vs_gt.py" \
      --cases-json "${cases}" --pred-dir "${pred}/pred" --gt-dir "${GT_DIR}" \
      --out-json "${detail}" --tag "fdg_test20_${mkey}" --workers 12
  python3 - <<PY
import json
from pathlib import Path
score = json.loads(Path("${detail}").read_text())
md = float(score.get("mean_dice_positive", score["mean_dice"]))
agg = {
    "method": "${mkey}",
    "mean_dice": md,
    "mean_dice_positive": md,
    "mean_dice_all_cases": score.get("mean_dice_all_cases"),
    "fp_rate": score.get("fp_rate"),
    "fn_rate": score.get("fn_rate"),
    "mean_fp": score.get("fp_rate"),
    "mean_fn": score.get("fn_rate"),
    "n_scored": score["n_scored"],
    "n_positive": score.get("n_positive"),
    "n_empty_gt": score.get("n_empty_gt"),
    "ckpt": "${ckpt}",
    "eval_stamp": "${EVAL_STAMP}",
}
Path("${out}/aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
PY
  _write_agg "${mkey}" "${out}/aggregate.json"
}

_eval_mae_family() {
  local method="$1" mkey="$2" depths="$3" use_v2="$4"
  local agg="${AGG_DIR}/aggregate_${mkey}.json"
  [[ "${SKIP_DONE}" == "1" && -f "${agg}" ]] && { echo "[fdg-test20] skip ${method} (aggregate exists)"; return 0; }
  _patch_board_running "${mkey}"
  local ckpt
  ckpt="$(_board_ckpt "${mkey}")"
  if [[ -z "${ckpt}" ]]; then
    if [[ "${method}" == "mae" ]]; then
      ckpt="${REPO}/runs/20260812_072719_iclr2026_mae_fdg_swinbase_gpu013_bs6_tr70_val10_100ep/best_seg_fdg_mae.pth"
    elif [[ "${method}" == "mae_scratch" || "${method}" == "monai_scratch" ]]; then
      ckpt=""
    else
      ckpt="${REPO}/runs/20260816_214921_iclr2026_monai_fdg_swinvit_1gpu_bs6_tr70_val10_100ep/best_seg_fdg_monai.pth"
    fi
  fi
  [[ -f "${ckpt}" ]] || { echo "[error] ${method} fdg ckpt missing: ${ckpt}" >&2; return 1; }
  local out="${EVAL_ROOT}/${method}"
  local out_json="${out}/test20.json"
  mkdir -p "${out}" "${CACHE}"
  docker run --rm \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    "${IMAGE_MAE}" \
    python3 "${CTRL}/ICLR2026/scripts/mae_preprocess_fdg_cache.py" \
      --cases-json "${TEST_JSON}" --images-tr "${RAW_MAE_IMG}" --labels-tr "${RAW_MAE_LAB}" \
      --out-dir "${CACHE}" --workers "${TASK1_MAE_PREP_WORKERS:-16}"
  echo "[fdg-test20] ${method} ckpt=${ckpt} gpu=${GPU}"
  docker run --rm --gpus "device=${GPU}" -e CUDA_VISIBLE_DEVICES=0 \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" -w "${REPO}" --shm-size=16g \
    "${IMAGE_MAE}" \
    python3 "${CTRL}/ICLR2026/scripts/mae_eval_seg_psma_test.py" \
      --cases-json "${TEST_JSON}" --cache-dir "${CACHE}" --seg-ckpt "${ckpt}" \
      --out-json "${out_json}" --depths "${depths}" --use-v2 "${use_v2}" \
      --tag "${method}_fdg_test20_${EVAL_STAMP}"
  python3 - <<PY
import json
from pathlib import Path
d = json.loads(Path("${out_json}").read_text())
md = float(d.get("mean_dice_positive", d["mean_dice"]))
agg = {
    "method": "${method}",
    "mean_dice": md,
    "mean_dice_positive": md,
    "mean_dice_all_cases": d.get("mean_dice_all_cases"),
    "fp_rate": d.get("fp_rate"),
    "fn_rate": d.get("fn_rate"),
    "mean_fp": d.get("fp_rate"),
    "mean_fn": d.get("fn_rate"),
    "n_scored": d.get("n_cases") or d.get("n_scored"),
    "n_positive": d.get("n_positive"),
    "n_empty_gt": d.get("n_empty_gt") or d.get("n_negative"),
    "ckpt": "${ckpt}",
    "eval_stamp": "${EVAL_STAMP}",
}
Path("${out}/aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
PY
  _write_agg "${mkey}" "${out}/aggregate.json"
}

_eval_dpdnet() {
  local mkey="${1:-dpdnet}"
  local agg="${AGG_DIR}/aggregate_${mkey}.json"
  [[ "${SKIP_DONE}" == "1" && -f "${agg}" ]] && { echo "[fdg-test20] skip dpdnet (aggregate exists)"; return 0; }
  _patch_board_running "${mkey}"
  local ckpt fdg_stamp tr
  ckpt="$(_board_ckpt "${mkey}")"
  fdg_stamp="$(_board_ckpt "${mkey}" stamp)"
  if [[ "${mkey}" == "dpdnet_dualenc" ]]; then tr="STUNetTrainer_small_prompt_pretrain"; else tr="STUNetTrainer_small_prompt"; fi
  [[ -z "${ckpt}" ]] && ckpt="${TASK1_DPDNET_FDG_BEST:-}"
  if [[ -z "${ckpt}" && "${mkey}" != "dpdnet_dualenc" ]]; then ckpt="/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/20260817_165250_iclr2026_dpdnet_fdg_2gpu_bs3_gbs6_n6_tr70_val0_169ep_gpu01/Dataset239_DpDNet_FDG_2ch/STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"; fi
  if [[ -z "${fdg_stamp}" && "${mkey}" == "dpdnet_dualenc" && -f "${VIS}/dpdnet_dualenc_fdg_LAST_STAMP.txt" ]]; then fdg_stamp="$(tr -d '[:space:]' < "${VIS}/dpdnet_dualenc_fdg_LAST_STAMP.txt")"; fi
  [[ -z "${fdg_stamp}" && -f "${VIS}/dpdnet_fdg_LAST_STAMP.txt" ]] && fdg_stamp="$(tr -d '[:space:]' < "${VIS}/dpdnet_fdg_LAST_STAMP.txt")"
  [[ -f "${ckpt}" ]] || { echo "[error] dpdnet fdg ckpt missing: ${ckpt}" >&2; return 1; }
  local out="${EVAL_ROOT}/${mkey}"
  local pred_out="${out}/predict"
  local detail="${out}/score_detail.json"
  local tmp_in="${out}/imagesTs_lymp"
  local tmp_pred="${out}/predict_lymp"
  local cases="${out}/cases_test_as_train.json"
  mkdir -p "${out}" "${pred_out}"
  _cases_json_train_only "${cases}"
  rm -rf "${tmp_in}" "${tmp_pred}" "${pred_out}"
  mkdir -p "${tmp_in}" "${tmp_pred}" "${pred_out}"
  chmod -R a+rwX "${out}" || true
  python3 - <<PY
import json, os
from pathlib import Path
raw = json.loads(Path("${TEST_JSON}").read_text())
cases = raw.get("cases") if isinstance(raw, dict) else list(raw)
pref = "lymp"
src = Path("${RAW_IMG}")
dst = Path("${tmp_in}")
for case in cases:
    for ch in (0, 1):
        s = src / f"{case}_{ch:04d}.nii.gz"
        d = dst / f"{pref}_{case}_{ch:04d}.nii.gz"
        if d.exists() or d.is_symlink():
            d.unlink()
        os.symlink(s, d)
print(f"[fdg-test20] dpdnet n={len(cases)}")
PY
  local ckpt_base
  ckpt_base="$(basename "${ckpt}")"
  echo "[fdg-test20] dpdnet fdg_stamp=${fdg_stamp} ckpt=${ckpt_base} gpu=${GPU}"
  docker run --rm --user root \
    --gpus "\"device=${GPU}\"" \
    -e CUDA_VISIBLE_DEVICES=0 -e HOME=/home/algorithm \
    -e nnUNet_raw="${WORK}/nnUNet_raw" \
    -e nnUNet_preprocessed="${WORK}/nnUNet_preprocessed" \
    -e nnUNet_results="${WORK}/nnUNet_results/${fdg_stamp}" \
    -e PYTHONPATH="${DPD}:/home/algorithm/.local/lib/python3.11/site-packages" \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" --shm-size=16g \
    --entrypoint bash "${IMAGE_NN}" -lc \
    "mkdir -p '${tmp_pred}' && python '${CTRL}/ICLR2026/scripts/dpdnet_predict_prompt_cli.py' -i '${tmp_in}' -o '${tmp_pred}' -d 239 -c 3d_fullres -tr ${tr} -f 0 -chk ${ckpt_base} -npp 2 -nps 2 --disable_tta"
  python3 - <<PY
from pathlib import Path
import shutil
pref = "lymp_"
src = Path("${tmp_pred}")
dst = Path("${pred_out}")
dst.mkdir(parents=True, exist_ok=True)
for p in src.glob("*.nii.gz"):
    name = p.name[len(pref):] if p.name.startswith(pref) else p.name
    shutil.copy2(p, dst / name)
print(f"[fdg-test20] dpdnet preds n={len(list(dst.glob('*.nii.gz')))}")
PY
  docker run --rm -v "${ROOT}:${ROOT}" -v "${DATA}:${DATA}" "${IMAGE_MAE}" \
    python3 "${ROOT}/ICLR2026/scripts/score_pred_dice_vs_gt.py" \
      --cases-json "${cases}" --pred-dir "${pred_out}" --gt-dir "${GT_DIR}" \
      --out-json "${detail}" --tag "fdg_test20_dpdnet" --workers 8
  python3 - <<PY
import json
from pathlib import Path
score = json.loads(Path("${detail}").read_text())
md = float(score.get("mean_dice_positive", score["mean_dice"]))
agg = {
    "method": "${mkey}",
    "mean_dice": md,
    "mean_dice_positive": md,
    "mean_dice_all_cases": score.get("mean_dice_all_cases"),
    "fp_rate": score.get("fp_rate"),
    "fn_rate": score.get("fn_rate"),
    "mean_fp": score.get("fp_rate"),
    "mean_fn": score.get("fn_rate"),
    "n_scored": score["n_scored"],
    "n_positive": score.get("n_positive"),
    "n_empty_gt": score.get("n_empty_gt"),
    "ckpt": "${ckpt}",
    "eval_stamp": "${EVAL_STAMP}",
}
Path("${out}/aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
PY
  _write_agg "${mkey}" "${out}/aggregate.json"
}

_eval_seganypet() {
  local mkey="${1:-seganypet}"
  local agg="${AGG_DIR}/aggregate_${mkey}.json"
  [[ "${SKIP_DONE}" == "1" && -f "${agg}" ]] && { echo "[fdg-test20] skip ${mkey} (aggregate exists)"; return 0; }
  _patch_board_running "${mkey}"
  local ckpt
  ckpt="$(_board_ckpt "${mkey}")"
  if [[ -z "${ckpt}" && "${mkey}" == "seganypet" ]]; then
    ckpt="${REPO}/runs/20260817_041526_iclr2026_seganypet_fdg_3gpu_bs6_gpu013/seganypet_fdg/best.pth"
  fi
  [[ -f "${ckpt}" ]] || { echo "[error] ${mkey} fdg ckpt missing: ${ckpt}" >&2; return 1; }
  local out="${EVAL_ROOT}/${mkey}"
  local out_json="${out}/test20.json"
  mkdir -p "${out}"
  python3 "${CTRL}/ICLR2026/scripts/prepare_seganypet_psma_test20.py" \
    --out-root "${SEG_TEST_ROOT}"
  echo "[fdg-test20] ${mkey} ckpt=${ckpt} gpu=${GPU}"
  docker run --rm --gpus "device=${GPU}" -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONPATH="${SEG_PIP}:${SEG_CODE}:${CTRL}/ICLR2026/scripts" \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" -w "${SEG_CODE}" --shm-size=8g \
    "${IMAGE_MAE}" \
    python3 "${CTRL}/ICLR2026/scripts/seganypet_eval_psma_test20_fold.py" \
      --ckpt "${ckpt}" --test-root "${SEG_TEST_ROOT}" \
      --pred-dir "${out}/pred" --out-json "${out_json}" \
      --fold 0 --stamp "${EVAL_STAMP}" --num-clicks "${TASK1_SEGANY_TEST_CLICKS:-5}"
  python3 - <<PY
import json
from pathlib import Path
d = json.loads(Path("${out_json}").read_text())
agg = {
    "method": "${mkey}",
    "mean_dice": d.get("mean_dice_positive", d.get("mean_dice")),
    "mean_dice_positive": d.get("mean_dice_positive", d.get("mean_dice")),
    "fp_rate": d.get("fp_rate"),
    "fn_rate": d.get("fn_rate"),
    "mean_fp": d.get("fp_rate", d.get("mean_fp")),
    "mean_fn": d.get("fn_rate", d.get("mean_fn")),
    "ckpt": "${ckpt}",
    "eval_stamp": "${EVAL_STAMP}",
    "protocol": d.get("protocol"),
}
Path("${out}/aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
PY
  _write_agg "${mkey}" "${out}/aggregate.json"
}

_eval_proto() {
  local agg="${AGG_DIR}/aggregate_proto_retrieval.json"
  [[ "${SKIP_DONE}" == "1" && -f "${agg}" ]] && { echo "[fdg-test20] skip proto (aggregate exists)"; return 0; }
  _patch_board_running proto_retrieval
  local strat="${CTRL}/ICLR2026/data/splits_stratified_70_10_20.json"
  local fdg_test_json="${TEST_JSON}"
  local gallery="${TASK1_PROTO_FD70_GALLERY:-${VIS}/proto_retrieval/fdg70_gallery.npz}"
  local out="${EVAL_ROOT}/proto_retrieval"
  local out_json="${out}/fdg_test20_all.json"
  mkdir -p "${out}" "$(dirname "${gallery}")"
  if [[ ! -f "${gallery}" ]]; then
    echo "[fdg-test20] building FDG70% gallery → ${gallery}"
    docker run --rm --user "$(id -u):$(id -g)" -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" -w "${CTRL}" "${IMAGE_MAE}" \
      python3 "${CTRL}/ICLR2026/scripts/proto_retrieval_build_gallery.py" \
        --stratified-json "${strat}" --pool fdg70 \
        --img-dir "${RAW_MAE_IMG}" --lab-dir "${RAW_MAE_LAB}" --out "${gallery}" \
        --workers "${TASK1_PROTO_GALLERY_WORKERS:-8}"
  fi
  echo "[fdg-test20] proto FDG70% sup → FDG20% TEST gallery=${gallery}"
  docker run --rm --user "$(id -u):$(id -g)" -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" -w "${CTRL}" "${IMAGE_MAE}" \
    python3 "${CTRL}/ICLR2026/scripts/proto_retrieval_eval_test20.py" \
      --gallery "${gallery}" --cases-json "${fdg_test_json}" \
      --img-dir "${RAW_MAE_IMG}" --lab-dir "${RAW_MAE_LAB}" \
      --pred-dir "${out}/predict" --out-json "${out_json}" \
      --fold 0 --topk "${TASK1_PROTO_TOPK:-3}" --stamp "${EVAL_STAMP}" \
      --pool-mode fdg70 \
      --tag "proto_fdg70_sup_fdg20_test"
  python3 - <<PY
import json
from pathlib import Path
d = json.loads(Path("${out_json}").read_text())
md = float(d.get("mean_dice_positive", d["mean_dice"]))
agg = {
    "method": "proto_retrieval",
    "mean_dice": md,
    "mean_dice_positive": md,
    "mean_dice_all_cases": d.get("mean_dice_all_cases"),
    "fp_rate": d.get("fp_rate"),
    "fn_rate": d.get("fn_rate"),
    "mean_fp": d.get("fp_rate"),
    "mean_fn": d.get("fn_rate"),
    "n_scored": d.get("n_scored") or d.get("n_cases"),
    "n_positive": d.get("n_positive"),
    "n_empty_gt": d.get("n_empty_gt"),
    "support_pool": "FDG70%",
    "test_split": "FDG20%",
    "topk": int("${TASK1_PROTO_TOPK:-3}"),
    "eval_stamp": "${EVAL_STAMP}",
    "protocol": "proto_retrieval_fdg70_sup_fdg20_test",
}
Path("${out}/aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
PY
  _write_agg proto_retrieval "${out}/aggregate.json"
}

_run() {
  case "$1" in
    nnunet) _eval_nnunet nnunet ;;
    nnunet_mim) _eval_nnunet nnunet_mim ;;
    mae) _eval_mae_family mae mae_swinunetr "2,2,6,2" 1 ;;
    mae_scratch) _eval_mae_family mae_scratch mae_scratch "2,2,6,2" 1 ;;
    monai) _eval_mae_family monai monai_swinvit "2,2,2,2" 0 ;;
    monai_scratch) _eval_mae_family monai_scratch monai_scratch "2,2,2,2" 0 ;;
    dpdnet) _eval_dpdnet dpdnet ;;
    dpdnet_dualenc) _eval_dpdnet dpdnet_dualenc ;;
    seganypet) _eval_seganypet seganypet ;;
    seganypet_scratch) _eval_seganypet seganypet_scratch ;;
    proto) _eval_proto ;;
    *) echo "[error] unknown method $1" >&2; return 1 ;;
  esac
}

if [[ "${METHOD}" == "all" ]]; then
  for m in nnunet proto mae monai dpdnet seganypet; do
    _run "${m}" || echo "[warn] ${m} failed" >&2
  done
else
  _run "${METHOD}"
fi

_patch_board_done
echo "[fdg-test20] DONE stamp=${EVAL_STAMP} aggregates=${AGG_DIR}"
echo "STAMP=${EVAL_STAMP}" > "${VIS}/fdg_test20_LAST_STAMP.txt"
