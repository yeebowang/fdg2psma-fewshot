#!/usr/bin/env bash
# Resume aligned pipeline AFTER MAE TEST20 is done: nnUNet bs=2 -> MONAI -> SegAnyPET.
# Do not re-wait MAE / re-run TEST20.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"
BOARD_PNG="${TASK1_ALIGN_BOARD_PNG:-${LOG_DIR}/progress_iclr2026_aligned_fdg_fs50_f258_board.png}"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"
export TASK1_BASE="${DATA}"

MAE_STAMP="${TASK1_MAE_WAIT_STAMP:-20260816_161859_iclr2026_mae_psma_fs50_from_fdg_seg_f258_gpu013}"
PIPE_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_aligned_resume_nnunet"
PIPE_LOG="${LOG_DIR}/nohup_aligned_resume_${PIPE_STAMP}.log"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[aligned-resume] PIPE=${PIPE_STAMP} mae_done=${MAE_STAMP}"

_board() {
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" --png "${BOARD_PNG}" --patch-json "$1" || true
}

# mark MAE TEST20 done on board (aggregate already written by eval script)
python3 - <<PY
import json
from pathlib import Path
board_p = Path("${BOARD_JSON}")
board = json.loads(board_p.read_text()) if board_p.is_file() else {}
agg_p = Path("${REPO}/runs/${MAE_STAMP}/psma_test20_eval/aggregate_test20_f258.json")
st = board.setdefault("methods", {}).setdefault("mae_swinunetr", {}).setdefault("psma_fs50_f258", {})
if agg_p.is_file():
    agg = json.loads(agg_p.read_text())
    st.update({
        "status": "done",
        "phase": None,
        "stamp": "${MAE_STAMP}",
        "fold_dice": agg.get("fold_test_dice") or {},
        "mean": agg.get("test_mean"),
        "metric": "TEST20 Dice (final); VAL10=monitor",
        "note": "TEST20 done",
        "eta": None,
        "eta_sec": None,
    })
    board["updated_note"] = f"MAE TEST20 mean={agg.get('test_mean')}"
board_p.write_text(json.dumps(board, indent=2) + "\n")
print("mae board", st.get("mean"))
PY
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD_JSON}" || true

# ---------- 1) nnUNet PSMA bs=2 ----------
NN_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_nnunet_psma_fs50_f258_1gpu_bs2_tr70_val70e20_300ep_gpu013"
echo "[aligned-resume] === PRIORITY nnUNet PSMA bs=2 ${NN_STAMP} ==="
_board "{\"queue\":[\"nnunet.psma_fs50_f258 bs=2 running\",\"monai\",\"seganypet\"],\"methods\":{\"nnunet\":{\"psma_fs50_f258\":{\"status\":\"running\",\"stamp\":\"${NN_STAMP}\",\"bs\":2,\"bs_note\":\"per-GPU\",\"total_epochs\":300,\"online_val\":\"VAL70 every20\",\"note\":\"PRIORITY re-run\"}}},\"updated_note\":\"nnUNet PSMA bs=2 running\"}"

TASK1_FIXED_BATCH_3D_FULLRES=2 \
TASK1_FS50_VAL_ITERS=70 \
TASK1_FS50_VAL_EVERY_N_EPOCHS=20 \
TASK1_NUM_EPOCHS=300 \
TASK1_TRAIN_ITERS_PER_EPOCH=70 \
TASK1_NNUNET_RESULTS_STAMP_NAME="${NN_STAMP}" \
  bash "${CTRL}/ICLR2026/run/run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh"

NN_AGG="${LOG_DIR}/aggregate_nnunet_psma_fs50_f258_${NN_STAMP}.json"
if [[ -f "${NN_AGG}" ]]; then
  python3 - <<PY
import json
from pathlib import Path
board_p = Path("${BOARD_JSON}")
board = json.loads(board_p.read_text())
agg = json.loads(Path("${NN_AGG}").read_text())
folds = {k: v.get("best_val_dice") for k, v in agg.get("folds", {}).items()}
st = board["methods"]["nnunet"]["psma_fs50_f258"]
st.update({
    "status": "done",
    "stamp": "${NN_STAMP}",
    "bs": 2,
    "bs_note": "per-GPU",
    "fold_dice": folds,
    "mean": agg.get("fold_mean"),
    "note": "re-run under FDG=6/PSMA=2 policy",
})
board["updated_note"] = "nnUNet PSMA bs=2 done"
board_p.write_text(json.dumps(board, indent=2) + "\n")
print("nnunet ingested", st.get("mean"))
PY
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD_JSON}" || true
fi

# ---------- 2) MONAI FDG → fewshot ----------
MONAI_FDG_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_monai_fdg_swinvit_gpu013_bs6_tr70_val10_100ep"
echo "[aligned-resume] === MONAI FDG ${MONAI_FDG_STAMP} ==="
_board "{\"updated_note\":\"queue: MONAI FDG\",\"queue\":[\"nnunet done\",\"monai_swinvit.fdg_pretrain running\",\"monai fs\",\"seganypet\"]}"
TASK1_NNUNET_RESULTS_STAMP_NAME="${MONAI_FDG_STAMP}" \
  bash "${CTRL}/ICLR2026/run/run_monai_fdg_swinbase_finetune_100ep_bg.sh"
MONAI_BEST="${REPO}/runs/${MONAI_FDG_STAMP}/best_seg_fdg_monai.pth"
[[ -f "${MONAI_BEST}" ]] || MONAI_BEST="${REPO}/runs/${MONAI_FDG_STAMP}/latest_seg_fdg_monai.pth"

MONAI_FS_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_monai_psma_fs50_from_fdg_seg_f258_gpu013"
TASK1_MONAI_FDG_SEG_CKPT="${MONAI_BEST}" \
TASK1_MAE_BATCH_SIZE=2 \
TASK1_NNUNET_RESULTS_STAMP_NAME="${MONAI_FS_STAMP}" \
  bash "${CTRL}/ICLR2026/run/run_monai_psma_fewshot50_f258_from_fdg_seg_bg.sh"
METHOD=monai STAMP="${MONAI_FS_STAMP}" \
  bash "${CTRL}/ICLR2026/run/run_eval_psma_test20_f258_bg.sh"

# ---------- 3) SegAnyPET FDG → fewshot ----------
SEG_FDG_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_fdg_pretrain_gpu013"
TASK1_SEGANY_FDG_BATCH_SIZE=6 \
TASK1_NNUNET_RESULTS_STAMP_NAME="${SEG_FDG_STAMP}" \
  bash "${CTRL}/ICLR2026/run/run_seganypet_fdg_pretrain_bg.sh"
SEG_BEST="${REPO}/runs/${SEG_FDG_STAMP}/seganypet_fdg/best.pth"
[[ -f "${SEG_BEST}" ]] || SEG_BEST="${REPO}/runs/${SEG_FDG_STAMP}/seganypet_fdg/latest.pth"

SEG_FS_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_fs50_from_fdg_f258_gpu013"
_board "{\"methods\":{\"seganypet\":{\"psma_fs50_f258\":{\"status\":\"running\",\"stamp\":\"${SEG_FS_STAMP}\",\"bs\":2,\"bs_note\":\"per-GPU\",\"foundation\":\"${SEG_BEST}\"}}},\"updated_note\":\"SegAnyPET fewshot bs=2\"}"
TASK1_SEGANY_CKPT="${SEG_BEST}" \
TASK1_SEGANY_BATCH_SIZE=2 \
TASK1_SEGANY_EPOCHS=100 \
TASK1_SEGANY_ACCUM=20 \
TASK1_SEGANY_LR_MODE=official \
TASK1_SEGANY_CLICK_MAX=21 \
TASK1_NNUNET_RESULTS_STAMP_NAME="${SEG_FS_STAMP}" \
  bash "${CTRL}/ICLR2026/run/run_seganypet_fewshot50_f258_bg.sh"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" --ingest-seganypet-stamp "${SEG_FS_STAMP}" || true

_board "{\"updated_note\":\"ALL STAGES DONE\",\"queue\":[]}"
echo "[aligned-resume] ALL DONE ${PIPE_STAMP}"
