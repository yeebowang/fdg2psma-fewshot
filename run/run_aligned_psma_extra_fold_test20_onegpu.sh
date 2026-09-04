#!/usr/bin/env bash
# Run one missing extra-fold TEST20 (folds 0/1/3/4/6/7) on a single idle GPU.
# Used by gpu-idle queue after extra-fold training is done.
#
#   bash ICLR2026/run/run_aligned_psma_extra_fold_test20_onegpu.sh --gpu 3
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${ROOT}/ICLR2026/vis"
REPO="${ROOT}/ICLR2026/3D-MAE-PET-CT"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
DONE_MARK="${VIS}/TASK1_PSMA_EXTRA_FOLD_TEST20_DONE.txt"

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

echo "[extra-fold-test20] gpu=${GPU} pick next missing TEST20 fold"
SKIP_FOLDS="${VIS}/TASK1_EXTRA_FOLD_TEST20_SKIPPED.txt"
INFLIGHT="${VIS}/TASK1_EXTRA_FOLD_TEST20_INFLIGHT.txt"
PICK_LOCK="${VIS}/TASK1_EXTRA_FOLD_TEST20_PICK.lock"

export EXTRA_FOLD_OWNER="$$"
JOB="$(
BOARD="${BOARD}" REPO="${REPO}" WORK="${WORK}" SKIP_FOLDS="${SKIP_FOLDS}" \
INFLIGHT="${INFLIGHT}" PICK_LOCK="${PICK_LOCK}" EXTRA_FOLD_OWNER="${EXTRA_FOLD_OWNER}" python3 - <<'PY'
import fcntl
import json
import os
from pathlib import Path

board = json.loads(Path(os.environ["BOARD"]).read_text(encoding="utf-8"))
methods = board.get("methods") or {}
repo = Path(os.environ["REPO"])
work = Path(os.environ["WORK"]) / "nnUNet_results"
extra = [0, 1, 3, 4, 6, 7]
order = [
    ("mae_swinunetr", "mae", (50, 10, 5)),
    ("monai_swinvit", "monai", (50, 10, 5)),
    ("nnunet", "nnunet", (50, 10, 5)),
    ("dpdnet", "dpdnet", (50, 10, 5)),
    ("seganypet", "seganypet", (50, 10, 5)),
]


def stamp_of(mkey, n):
    st = (methods.get(mkey) or {}).get(f"psma_fs{n}_f258") or {}
    return (st.get("stamp") or "").strip()


def has_ckpt(short, stamp, n, fold):
    if not stamp:
        return False
    if short == "mae":
        d = repo / "runs" / stamp / "mae" / f"fold{fold}"
        return any(d.glob("best_*.pth")) or any(d.glob("latest_*.pth"))
    if short == "monai":
        d = repo / "runs" / stamp / "monai" / f"fold{fold}"
        return any(d.glob("best_*.pth")) or any(d.glob("latest_*.pth"))
    if short == "seganypet":
        d = repo / "runs" / stamp / "seganypet" / f"fold{fold}"
        return any(d.glob("*.pth")) or (d / "best.pth").is_file()
    for cand in (work / f"{stamp}_f{fold}", work / stamp):
        if any(cand.glob(f"**/fold_{fold}/checkpoint_*.pth")):
            return True
        if cand.name.endswith(f"_f{fold}") and any(cand.glob("**/checkpoint_*.pth")):
            return True
    return False


def has_test20(short, stamp, fold):
    if short in ("mae", "monai", "seganypet"):
        return (repo / "runs" / stamp / "psma_test20_eval" / f"fold{fold}_test20.json").is_file()
    return (work / stamp / "psma_test20_eval" / f"fold{fold}" / "score_detail.json").is_file()


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
with lock_p.open("a+") as lf:
    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
    skip_p = Path(os.environ.get("SKIP_FOLDS") or "")
    skipped: set[str] = set()
    if skip_p.is_file():
        skipped = {ln.strip() for ln in skip_p.read_text(encoding="utf-8").splitlines() if ln.strip()}
    inflight_p = Path(os.environ["INFLIGHT"])
    inflight = load_inflight(inflight_p)
    try:
        owner = int(os.environ.get("EXTRA_FOLD_OWNER") or "0")
    except ValueError:
        owner = 0
    if owner <= 0:
        owner = os.getppid()
    for mkey, short, ns in order:
        for n in ns:
            stamp = stamp_of(mkey, n)
            if not stamp:
                continue
            for fold in extra:
                key = f"{short}|{n}|{fold}"
                if key in skipped or key in inflight:
                    continue
                if has_test20(short, stamp, fold):
                    continue
                if not has_ckpt(short, stamp, n, fold):
                    continue
                inflight[key] = owner
                save_inflight(inflight_p, inflight)
                print(f"{short}|{n}|{fold}|{stamp}")
                raise SystemExit
    save_inflight(inflight_p, inflight)
print("")
PY
)"

if [[ -z "${JOB}" ]]; then
  echo "[extra-fold-test20] no missing extra-fold TEST20 — mark done"
  {
    echo "done_at=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "status=ok"
    echo "note=all extra-fold TEST20 present"
  } > "${DONE_MARK}"
  exit 0
fi

IFS='|' read -r METH N FOLD STAMP <<< "${JOB}"
echo "[extra-fold-test20] job method=${METH} fs${N} fold=${FOLD} stamp=${STAMP} gpu=${GPU}"

_drop_inflight() {
  [[ -n "${METH:-}" && -n "${N:-}" && -n "${FOLD:-}" ]] || return 0
  INFLIGHT="${INFLIGHT}" PICK_LOCK="${PICK_LOCK}" KEY="${METH}|${N}|${FOLD}" python3 - <<'PY' || true
import fcntl, os
from pathlib import Path
key = os.environ.get("KEY") or ""
lock_p = Path(os.environ["PICK_LOCK"])
inf_p = Path(os.environ["INFLIGHT"])
lock_p.touch(exist_ok=True)
with lock_p.open("a+") as lf:
    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
    if not inf_p.is_file() or not key:
        raise SystemExit
    keep = []
    for ln in inf_p.read_text(encoding="utf-8").splitlines():
        if ln.strip() and not ln.startswith(key + "\t") and ln.strip() != key:
            keep.append(ln)
    inf_p.write_text(("\n".join(keep) + ("\n" if keep else "")), encoding="utf-8")
PY
}
trap _drop_inflight EXIT

STAGE="psma_fs${N}_f258"
case "${METH}" in
  mae) BOARD_KEY="mae_swinunetr" ;;
  monai) BOARD_KEY="monai_swinvit" ;;
  *) BOARD_KEY="${METH}" ;;
esac

_fail_skip() {
  # ckpt 已在：predict 可 resume。写入 skip 会让板上永久空着（nnUNet fs50 f6/f7）。
  echo "[extra-fold-test20] FAIL retry later ${METH} fs${N} f${FOLD} gpu=${GPU}" >&2
  python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD}" --no-plot \
    --patch-json "{\"updated_note\":\"extra-fold TEST20 retry ${METH} fs${N} f${FOLD}\"}" || true
  exit 0
}

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"${BOARD_KEY}\":{\"${STAGE}\":{\"status\":\"running\",\"note\":\"9fold TEST20 · ${METH} fs${N} f${FOLD} · GPU ${GPU}\"}}},\"updated_note\":\"gpu-idle extra-fold TEST20 ${METH} fs${N} f${FOLD} GPU ${GPU}\"}" || true

rc=0
set +e
case "${METH}" in
  mae|monai)
    METHOD="${METH}" STAMP="${STAMP}" \
    TASK1_FEWSHOT_N="${N}" TASK1_PSMA_BOARD_STAGE="${STAGE}" \
    TASK1_MAE_FEWSHOT_FOLDS_CSV="${FOLD}" \
    TASK1_FOLD_GPUS="${FOLD}:${GPU}" \
    TASK1_TEST_SKIP_DONE=1 \
    TASK1_CUDA_VISIBLE_DEVICES="${GPU}" \
    TASK1_UDA_PRED_PER_GPU=1 \
      bash "${ROOT}/ICLR2026/run/run_eval_psma_test20_f258_bg.sh"
    rc=$?
    ;;
  seganypet)
    STAMP="${STAMP}" TASK1_FEWSHOT_N="${N}" TASK1_PSMA_BOARD_STAGE="${STAGE}" \
    TASK1_SEGANY_FOLDS_CSV="${FOLD}" \
    TASK1_FOLD_GPUS="${FOLD}:${GPU}" \
    TASK1_TEST_SKIP_DONE=1 \
    TASK1_CUDA_VISIBLE_DEVICES="${GPU}" \
      bash "${ROOT}/ICLR2026/run/run_eval_seganypet_psma_test20_f258_bg.sh"
    rc=$?
    ;;
  nnunet)
    PARENT_STAMP="${STAMP}" TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" \
    TASK1_FEWSHOT_N="${N}" TASK1_PSMA_BOARD_STAGE="${STAGE}" \
    TASK1_FOLDS="${FOLD}" TASK1_FOLD_GPUS="${FOLD}:${GPU}" \
    TASK1_TEST_SKIP_DONE=1 \
    TASK1_CUDA_VISIBLE_DEVICES="${GPU}" \
    TASK1_UDA_PRED_PER_GPU=1 \
      bash "${ROOT}/ICLR2026/run/run_nnunet_psma_test20_f258_parallel.sh"
    rc=$?
    ;;
  dpdnet)
    PARENT_STAMP="${STAMP}" TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" \
    TASK1_FEWSHOT_N="${N}" TASK1_PSMA_BOARD_STAGE="${STAGE}" \
    TASK1_FOLDS="${FOLD}" TASK1_FOLD_GPUS="${FOLD}:${GPU}" \
    TASK1_TEST_SKIP_DONE=1 \
    TASK1_CUDA_VISIBLE_DEVICES="${GPU}" \
    TASK1_UDA_PRED_PER_GPU=1 \
      bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_test20_f258_parallel.sh"
    rc=$?
    ;;
  *)
    echo "[error] unknown method ${METH}" >&2
    exit 2
    ;;
esac
set -e
[[ "${rc}" -eq 0 ]] || _fail_skip

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot || true

echo "[extra-fold-test20] finished ${METH} fs${N} f${FOLD} gpu=${GPU}"
