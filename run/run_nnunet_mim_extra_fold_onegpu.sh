#!/usr/bin/env bash
# Train the next missing nnUNet MIM extra fold (0,1,3,4,6,7) on one GPU.
#   bash ICLR2026/run/run_nnunet_mim_extra_fold_onegpu.sh --gpu 1
# Exit 0 with "no missing" when done; exit 0 after one fold trained.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
INFLIGHT="${VIS}/TASK1_NNUNET_MIM_EXTRA_FOLD_INFLIGHT.txt"
PICK_LOCK="${VIS}/TASK1_NNUNET_MIM_EXTRA_FOLD_PICK.lock"

GPU="${TASK1_EXTRA_FOLD_GPU:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:?}"; shift 2 ;;
    *) shift ;;
  esac
done
[[ -n "${GPU}" ]] || { echo "[error] need --gpu N" >&2; exit 2; }
[[ "${GPU}" != "2" ]] || { echo "[error] GPU2 reserved" >&2; exit 2; }

MIM_FDG_STAMP="${TASK1_UDA_FDG_STAMP:-20260829_133121_iclr2026_nnunet_mim_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep}"
MIM_FDG_BEST="${TASK1_UDA_FDG_BEST:-${WORK}/nnUNet_results/${MIM_FDG_STAMP}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth}"
[[ -f "${MIM_FDG_BEST}" ]] || { echo "[error] missing MIM FDG ckpt ${MIM_FDG_BEST}" >&2; exit 1; }

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD}"
export TASK1_BOARD_METHOD=nnunet_mim

JOB="$(
BOARD="${BOARD}" WORK="${WORK}" INFLIGHT="${INFLIGHT}" PICK_LOCK="${PICK_LOCK}" \
EXTRA_FOLD_OWNER="$$" python3 - <<'PY'
import fcntl, json, os
from pathlib import Path

board = json.loads(Path(os.environ["BOARD"]).read_text(encoding="utf-8"))
work = Path(os.environ["WORK"]) / "nnUNet_results"
methods = board.get("methods") or {}
extra = [0, 1, 3, 4, 6, 7]
order = [50, 10, 5]

def stamp_of(n: int) -> str:
    st = (methods.get("nnunet_mim") or {}).get(f"psma_fs{n}_f258") or {}
    return (st.get("stamp") or "").strip()

def has_fold(stamp: str, fold: int) -> bool:
    if not stamp:
        return False
    for cand in (work / f"{stamp}_f{fold}", work / stamp):
        if any(cand.glob(f"**/fold_{fold}/checkpoint_*.pth")):
            return True
        if cand.name.endswith(f"_f{fold}") and any(cand.glob("**/checkpoint_*.pth")):
            return True
    return False

def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True

def load_inflight(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.is_file():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        key = parts[0].strip()
        try:
            pid = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            pid = 0
        if key and pid_alive(pid):
            out[key] = pid
    return out

def save_inflight(path: Path, rec: dict[str, int]) -> None:
    lines = [f"{k}\t{pid}" for k, pid in sorted(rec.items())]
    path.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")

lock_p = Path(os.environ["PICK_LOCK"])
lock_p.parent.mkdir(parents=True, exist_ok=True)
lock_p.touch(exist_ok=True)
owner = int(os.environ.get("EXTRA_FOLD_OWNER") or "0") or os.getppid()
with lock_p.open("a+") as lf:
    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
    inflight_p = Path(os.environ["INFLIGHT"])
    inflight = load_inflight(inflight_p)
    for n in order:
        stamp = stamp_of(n)
        if not stamp:
            continue
        for fold in extra:
            key = f"nnunet_mim|{n}|{fold}"
            if key in inflight:
                continue
            if has_fold(stamp, fold):
                continue
            inflight[key] = owner
            save_inflight(inflight_p, inflight)
            print(f"{n}|{fold}|{stamp}")
            raise SystemExit
    save_inflight(inflight_p, inflight)
print("")
PY
)"

if [[ -z "${JOB}" ]]; then
  echo "[mim-extra] no missing folds"
  exit 0
fi

IFS='|' read -r N FOLD STAMP <<< "${JOB}"
echo "[mim-extra] fs${N} fold=${FOLD} stamp=${STAMP} gpu=${GPU}"

_drop_inflight() {
  INFLIGHT="${INFLIGHT}" PICK_LOCK="${PICK_LOCK}" KEY="nnunet_mim|${N}|${FOLD}" python3 - <<'PY' || true
import fcntl, os
from pathlib import Path
key = os.environ.get("KEY") or ""
lock_p = Path(os.environ["PICK_LOCK"]); inf_p = Path(os.environ["INFLIGHT"])
lock_p.touch(exist_ok=True)
with lock_p.open("a+") as lf:
    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
    if not inf_p.is_file() or not key:
        raise SystemExit
    keep = [ln for ln in inf_p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith(key + "\t") and ln.strip() != key]
    inf_p.write_text(("\n".join(keep) + ("\n" if keep else "")), encoding="utf-8")
PY
}
trap _drop_inflight EXIT

SPLIT_DIR="${ROOT}/ICLR2026/data/splits_mae_psma_fewshot${N}_9fold"
[[ -f "${SPLIT_DIR}/fold0_nnunet.json" ]] || python3 "${ROOT}/ICLR2026/scripts/export_mae_psma_fewshot50_9fold.py" \
  --n-shot "${N}" --out-dir "${SPLIT_DIR}" --seed 42

STAGE="psma_fs${N}_f258"
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"nnunet_mim\":{\"${STAGE}\":{\"status\":\"running\",\"note\":\"9fold extra · MIM fs${N} f${FOLD} · GPU ${GPU}\"}}},\"updated_note\":\"nnUNet MIM extra fold fs${N} f${FOLD} GPU ${GPU}\"}" || true

FOLD_ID="${FOLD}" GPU_ID="${GPU}" PARENT_STAMP="${STAMP}" \
TASK1_FEWSHOT_N="${N}" \
TASK1_FEWSHOT_SPLIT_DIR="${SPLIT_DIR}" \
TASK1_DOCKER_BACKGROUND=0 \
TASK1_NUM_EPOCHS=100 \
TASK1_TRAIN_ITERS_PER_EPOCH=25 \
TASK1_VAL_ITERS_PER_EPOCH=25 \
TASK1_VAL_EVERY_N_EPOCHS=20 \
TASK1_FS50_VAL_ITERS=25 \
TASK1_FS50_VAL_EVERY_N_EPOCHS=20 \
TASK1_FIXED_BATCH_3D_FULLRES=2 \
TASK1_BEST_BY=val_loss TASK1_VAL_LOSS_ONLY=1 \
TASK1_UDA_FDG_STAMP="${MIM_FDG_STAMP}" \
TASK1_UDA_FDG_BEST="${MIM_FDG_BEST}" \
TASK1_BOARD_METHOD=nnunet_mim \
  bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fewshot50_onefold_bg.sh"

echo "[mim-extra] finished fs${N} f${FOLD} gpu=${GPU}"
