#!/usr/bin/env bash
# Wait until nnUNet TEST20 score_detail exists for folds 2/5/8, aggregate, then start MONAI stage.
set -euo pipefail
PARENT="${PARENT_STAMP:?set PARENT_STAMP}"
EVAL="/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/${PARENT}/psma_test20_eval"
CTRL=/media/ybwang/data1/PSMA-CTRL
VIS=$CTRL/ICLR2026/vis
echo "[wait] need score_detail for f2/f5/f8 under ${EVAL}"
for f in 2 5 8; do
  while [[ ! -f "${EVAL}/fold${f}/score_detail.json" ]]; do
    echo "[wait] missing fold${f} $(date +%H:%M:%S)"
    sleep 45
  done
  echo "[wait] fold${f} ready"
done
export TASK1_BASE=/media/ybwang/data1/PSMA-DATA
export PARENT_STAMP="${PARENT}"
export TASK1_UDA_PRED_PER_GPU=5
export TASK1_TEST_SKIP_DONE=1
export TASK1_FOLDS=2,5,8
bash "${CTRL}/ICLR2026/run/run_nnunet_psma_test20_f258_parallel.sh"
AGG="/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/${PARENT}/aggregate_test20_dice_f258.json"
cp -f "${AGG}" "${VIS}/aggregate_nnunet_psma_fs50_f258_${PARENT}.json"
python3 - <<PY
import json
from pathlib import Path
from datetime import datetime
parent = "${PARENT}"
vis = Path("${VIS}")
agg = json.loads((Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results") / parent / "aggregate_test20_dice_f258.json").read_text())
board_p = vis / "iclr2026_aligned_fdg_fs50_f258_board.json"
board = json.loads(board_p.read_text())
st = board["methods"]["nnunet"]["psma_fs50_f258"]
fd = {k: v.get("test_dice") for k, v in (agg.get("folds") or {}).items()}
st.update({
    "status": "done", "stamp": parent, "fold_dice": fd, "mean": agg.get("fold_mean"),
    "phase": None, "note": "TEST20 done", "metric": "TEST20 Dice (final); VAL10=monitor",
    "eta": None, "eta_sec": None,
})
board["updated_note"] = f"nnUNet TEST20 mean={agg.get('fold_mean')}"
board["updated_at"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
board_p.write_text(json.dumps(board, indent=2) + "\n")
print("ingested", st["mean"], fd)
PY
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --plot-only || true
export FROM_STAGE=monai
bash "${CTRL}/ICLR2026/run/run_aligned_pipeline_stages_bg.sh"
