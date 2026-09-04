#!/usr/bin/env bash
# nnUNet-MIM remaining PSMA fewshot (fs10 then fs5) · one fold on one idle GPU.
# Used by gpu-idle queue. Does not touch the in-flight fs50 TEST20.
#
#   bash ICLR2026/run/run_nnunet_mim_remaining_onegpu.sh --gpu 3
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
PICK_LOCK="${VIS}/TASK1_NNUNET_MIM_REMAINING_PICK.lock"
INFLIGHT="${VIS}/TASK1_NNUNET_MIM_REMAINING_INFLIGHT.txt"

GPU="${TASK1_EXTRA_FOLD_GPU:-${TASK1_CUDA_VISIBLE_DEVICES:-}}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:?}"; shift 2 ;;
    *) shift ;;
  esac
done
GPU="${GPU%%,*}"
[[ -n "${GPU}" ]] || { echo "[error] need --gpu N" >&2; exit 2; }

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD}"
export TASK1_BOARD_METHOD=nnunet_mim

JOB="$(
BOARD="${BOARD}" WORK="${WORK}" VIS="${VIS}" INFLIGHT="${INFLIGHT}" PICK_LOCK="${PICK_LOCK}" python3 - <<'PY'
import fcntl
import json
import os
from pathlib import Path

board = json.loads(Path(os.environ["BOARD"]).read_text())
work = Path(os.environ["WORK"]) / "nnUNet_results"
vis = Path(os.environ["VIS"])
inflight_p = Path(os.environ["INFLIGHT"])
lock_p = Path(os.environ["PICK_LOCK"])
mim = (board.get("methods") or {}).get("nnunet_mim") or {}
fdg = mim.get("fdg_pretrain") or {}
fdg_stamp = str(fdg.get("stamp") or "").strip()
fdg_ckpt = str(fdg.get("best_ckpt") or "").strip()
if not fdg_stamp:
    raise SystemExit(0)
if not fdg_ckpt:
    fold0 = work / fdg_stamp / "Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0"
    for name in ("checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth"):
        p = fold0 / name
        if p.is_file():
            fdg_ckpt = str(p)
            break

lock_p.parent.mkdir(parents=True, exist_ok=True)
with lock_p.open("a+") as lf:
    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
    busy = set()
    if inflight_p.is_file():
        busy = {ln.strip() for ln in inflight_p.read_text().splitlines() if ln.strip()}

    def has_ckpt(parent: str, fold: int) -> bool:
        if not parent:
            return False
        for cand in (work / f"{parent}_f{fold}", work / parent):
            if any(cand.glob("**/checkpoint_*.pth")):
                return True
        return False

    for n in (10, 5):
        stage = f"psma_fs{n}_f258"
        st = mim.get(stage) or {}
        md = st.get("mean")
        if isinstance(md, (int, float)) and md == md:
            continue
        stamp = str(st.get("stamp") or "").strip()
        if not stamp:
            stamp = ""
        for fold in (2, 5, 8):
            key = f"{n}|{fold}"
            if key in busy:
                continue
            if stamp and has_ckpt(stamp, fold):
                continue
            busy.add(key)
            inflight_p.write_text("\n".join(sorted(busy)) + "\n")
            print(f"{n}|{fold}|{stamp}|{fdg_stamp}|{fdg_ckpt}")
            raise SystemExit
PY
)"
[[ -n "${JOB}" ]] || { echo "[nnunet-mim-1gpu] nothing pending"; exit 0; }
IFS='|' read -r FEW FOLD STAMP FDG_STAMP FDG_CKPT <<<"${JOB}"
STAGE="psma_fs${FEW}_f258"
if [[ -z "${STAMP}" ]]; then
  STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_nnunet_mim_psma_fs${FEW}_f258_1gpu_bs2_tr25_val25e20_100ep_gpu013"
fi
FOLD_STAMP="${STAMP}_f${FOLD}"

echo "[nnunet-mim-1gpu] gpu=${GPU} fs${FEW} fold${FOLD} parent=${STAMP} fold_stamp=${FOLD_STAMP}"

export TASK1_NNUNET_RESULTS_STAMP_NAME="${FOLD_STAMP}"
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"nnunet_mim\":{\"${STAGE}\":{\"status\":\"running\",\"stamp\":\"${STAMP}\",\"note\":\"fs${FEW} f${FOLD} · GPU ${GPU}\"}}},\"updated_note\":\"nnUNet MIM fs${FEW} f${FOLD} GPU ${GPU}\"}" || true

export TASK1_GUARD_STAMP="${FOLD_STAMP}"
export TASK1_GUARD_DATASET_DIR="Dataset228_AutoPETIV_Task1_2ch"
export TASK1_GUARD_TRAINER_FOLDER="nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres"
export TASK1_NUM_EPOCHS=100
export TASK1_GUARD_TOTAL_EPOCHS=100
export TASK1_GUARD_RESTART_SCRIPT="ICLR2026/run/run_nnunet_psma_fewshot50_onefold_bg.sh"
export TASK1_GUARD_EXTRA_ENV="FOLD_ID=${FOLD},GPU_ID=${GPU},PARENT_STAMP=${STAMP},TASK1_FEWSHOT_N=${FEW},TASK1_FEWSHOT_SPLIT_DIR=${CTRL}/ICLR2026/data/splits_mae_psma_fewshot${FEW}_9fold,TASK1_BOARD_METHOD=nnunet_mim,TASK1_UDA_FDG_STAMP=${FDG_STAMP},TASK1_UDA_FDG_BEST=${FDG_CKPT},TASK1_NUM_EPOCHS=100,TASK1_TRAIN_ITERS_PER_EPOCH=25,TASK1_VAL_ITERS_PER_EPOCH=25,TASK1_VAL_EVERY_N_EPOCHS=20,TASK1_FIXED_BATCH_3D_FULLRES=2,TASK1_BEST_BY=val_loss,TASK1_VAL_LOSS_ONLY=1"
bash "${CTRL}/run_task/run_task1_train_auto_resume_guard_bg.sh" || true

set +e
FOLD_ID="${FOLD}" GPU_ID="${GPU}" PARENT_STAMP="${STAMP}" \
  TASK1_FEWSHOT_N="${FEW}" \
  TASK1_FEWSHOT_SPLIT_DIR="${CTRL}/ICLR2026/data/splits_mae_psma_fewshot${FEW}_9fold" \
  TASK1_BOARD_METHOD=nnunet_mim \
  TASK1_UDA_FDG_STAMP="${FDG_STAMP}" \
  TASK1_UDA_FDG_BEST="${FDG_CKPT}" \
  TASK1_NUM_EPOCHS=100 \
  TASK1_TRAIN_ITERS_PER_EPOCH=25 \
  TASK1_VAL_ITERS_PER_EPOCH=25 \
  TASK1_VAL_EVERY_N_EPOCHS=20 \
  TASK1_FS50_VAL_ITERS=25 \
  TASK1_FS50_VAL_EVERY_N_EPOCHS=20 \
  TASK1_FIXED_BATCH_3D_FULLRES=2 \
  TASK1_BEST_BY=val_loss \
  TASK1_VAL_LOSS_ONLY=1 \
  TASK1_DOCKER_BACKGROUND=0 \
  bash "${CTRL}/ICLR2026/run/run_nnunet_psma_fewshot50_onefold_bg.sh"
rc=$?
set -e

python3 - <<PY
from pathlib import Path
p = Path("${INFLIGHT}")
key = "${FEW}|${FOLD}"
if p.is_file():
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and ln.strip() != key]
    p.write_text(("\n".join(lines) + "\n") if lines else "")
PY

export TASK1_NNUNET_RESULTS_STAMP_NAME="${FOLD_STAMP}"
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true
echo "[nnunet-mim-1gpu] fold done fs${FEW} f${FOLD} rc=${rc}"
exit "${rc}"
