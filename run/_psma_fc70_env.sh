# Shared PSMA fc70% (421/59 train70) protocol — source from run scripts.
# tr25 val25 bs2 · val every 20ep · single run (no f258) · FDG init · decline → TEST20
PSMA_FC70_STAGE="${TASK1_PSMA_BOARD_STAGE:-psma_fc70}"
PSMA_FC70_SPLITS="${TASK1_PSMA_FC70_SPLITS:-${ROOT}/ICLR2026/data/splits_baseline2_psma_uda_nnunet.json}"
PSMA_FC70_BS="${TASK1_FIXED_BATCH_3D_FULLRES:-2}"
PSMA_FC70_TR="${TASK1_TRAIN_ITERS_PER_EPOCH:-25}"
PSMA_FC70_VAL="${TASK1_FS50_VAL_ITERS:-25}"
PSMA_FC70_VAL_EVERY="${TASK1_FS50_VAL_EVERY_N_EPOCHS:-20}"
PSMA_FC70_EP="${TASK1_NUM_EPOCHS:-100}"
PSMA_FC70_MAX_EP="${TASK1_RESUME_MAX_EPOCHS:-300}"
PSMA_FC70_BASE_EP="${TASK1_RESUME_BASE_EP:-100}"
PSMA_FC70_GPU="${TASK1_PSMA_FC70_GPU:-0}"
PSMA_FC70_FOLD="${TASK1_PSMA_FC70_FOLD:-0}"

_fc70_resolve_iters() {
  python3 - <<PY
import json, math
from pathlib import Path
d = json.loads(Path("${PSMA_FC70_SPLITS}").read_text())[0]
bs = int("${PSMA_FC70_BS}")
ntr, nva = len(d["train"]), len(d["val"])
tr = int("${PSMA_FC70_TR}")
va = int("${PSMA_FC70_VAL}")
print(tr, va, ntr, nva)
PY
}
