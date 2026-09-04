#!/usr/bin/env bash
# MAE/MONAI PSMA fc70 TEST20 — single fold, one Dice → board mean.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"

METHOD="${METHOD:-mae}"
STAMP="${STAMP:?set STAMP}"
BOARD_STAGE="${TASK1_PSMA_BOARD_STAGE:-psma_fc70}"
FOLD="${TASK1_PSMA_FC70_FOLD:-0}"
GPU="${TASK1_PSMA_FC70_GPU:-0}"
TEST_JSON="${ROOT}/ICLR2026/data/splits_mae_psma_test20.json"
CACHE="${DATA}/task1_train_workspace/mae_cache/psma_baseline2_70_10"

if [[ "${METHOD}" == "mae" ]]; then
  SUB=mae; STEM="seg_psma_fc70_fdgseg_f${FOLD}"; BOARD_KEY=mae_swinunetr; DEPTHS=2,2,6,2; USE_V2=1
elif [[ "${METHOD}" == "mae_scratch" ]]; then
  SUB=mae; STEM="seg_psma_fc70_fdgseg_f${FOLD}"; BOARD_KEY=mae_scratch; DEPTHS=2,2,6,2; USE_V2=1
elif [[ "${METHOD}" == "monai_scratch" ]]; then
  SUB=monai; STEM="seg_psma_fc70_monai_fdgseg_f${FOLD}"; BOARD_KEY=monai_scratch; DEPTHS=2,2,2,2; USE_V2=0
else
  SUB=monai; STEM="seg_psma_fc70_monai_fdgseg_f${FOLD}"; BOARD_KEY=monai_swinvit; DEPTHS=2,2,2,2; USE_V2=0
fi

OUT_ROOT="${REPO}/runs/${STAMP}"
EVAL_ROOT="${OUT_ROOT}/psma_test20_eval"
mkdir -p "${EVAL_ROOT}" "${CACHE}"
ckpt="${OUT_ROOT}/${SUB}/fold${FOLD}/best_${STEM}.pth"
[[ -f "${ckpt}" ]] || ckpt="${OUT_ROOT}/${SUB}/fold${FOLD}/latest_${STEM}.pth"
[[ -f "${ckpt}" ]] || { echo "[error] missing ${ckpt}" >&2; exit 1; }

out_json="${EVAL_ROOT}/fold${FOLD}_test20.json"
docker run --rm --gpus "device=${GPU}" -e CUDA_VISIBLE_DEVICES=0 \
  -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" -w "${REPO}" --shm-size=16g "${IMAGE}" \
  python3 "${CTRL}/ICLR2026/scripts/mae_eval_seg_psma_test.py" \
    --cases-json "${TEST_JSON}" --cache-dir "${CACHE}" --seg-ckpt "${ckpt}" \
    --out-json "${out_json}" --depths "${DEPTHS}" --use-v2 "${USE_V2}" \
    --tag "${METHOD}_fc70_${STAMP}"

python3 - <<PY
import json
from pathlib import Path
d = json.loads(Path("${out_json}").read_text())
md = float(d.get("mean_dice_positive", d.get("mean_dice")))
fp = d.get("fp_rate", d.get("mean_fp"))
fn = d.get("fn_rate", d.get("mean_fn"))
summary = {
    "stamp": "${STAMP}",
    "method": "${METHOD}",
    "single_run": True,
    "fold_test_dice": {"${FOLD}": md},
    "test_mean": md,
    "mean_dice": md,
    "mean_dice_positive": d.get("mean_dice_positive", md),
    "fp_rate": fp,
    "fn_rate": fn,
    "mean_fp": fp,
    "mean_fn": fn,
    "test_std": 0.0,
}
Path("${EVAL_ROOT}/aggregate_test20_fc70.json").write_text(json.dumps(summary, indent=2) + "\n")
Path("${LOG_DIR}/aggregate_${METHOD}_psma_fc70_${STAMP}.json").write_text(json.dumps(summary, indent=2) + "\n")
board = json.loads(Path("${BOARD}").read_text())
st = board["methods"]["${BOARD_KEY}"]["${BOARD_STAGE}"]
st.update({
    "status": "done",
    "stamp": "${STAMP}",
    "fold_dice": {"${FOLD}": md},
    "mean": md,
    "mean_fp": float(fp) if isinstance(fp, (int, float)) and fp == fp else None,
    "mean_fn": float(fn) if isinstance(fn, (int, float)) and fn == fn else None,
    "note": "TEST20 DONE · fc70 single",
    "metric": "TEST20 Dice/FP/FN; single run",
})
board["updated_note"] = f"${METHOD} fc70 TEST20 Dice={md:.4f} FP={fp} FN={fn}"
Path("${BOARD}").write_text(json.dumps(board, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" --no-plot || true
echo "[test20-fc70] ${METHOD} DONE"
