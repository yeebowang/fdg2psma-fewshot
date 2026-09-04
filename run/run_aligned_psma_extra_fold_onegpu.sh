#!/usr/bin/env bash
# Train one missing extra fold (0,1,3,4,6,7) of fs50/fs10/fs5 on a single idle GPU.
# Used by gpu-idle queue. Does not wait for fc70.
#
#   bash ICLR2026/run/run_aligned_psma_extra_fold_onegpu.sh --gpu 0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${ROOT}/ICLR2026/vis"
REPO="${ROOT}/ICLR2026/3D-MAE-PET-CT"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
DONE_MARK="${VIS}/TASK1_PSMA_EXTRA_FOLDS_9FOLD_DONE.txt"

GPU="${TASK1_EXTRA_FOLD_GPU:-${TASK1_CUDA_VISIBLE_DEVICES:-}}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:?}"; shift 2 ;;
    *) shift ;;
  esac
done
[[ -n "${GPU}" ]] || { echo "[error] need --gpu N" >&2; exit 2; }

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD}"

echo "[extra-fold-1gpu] gpu=${GPU} pick next missing fold"
SKIP_FOLDS="${VIS}/TASK1_EXTRA_FOLD_SKIPPED.txt"
INFLIGHT="${VIS}/TASK1_EXTRA_FOLD_INFLIGHT.txt"
PICK_LOCK="${VIS}/TASK1_EXTRA_FOLD_PICK.lock"

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


def has_mae(stamp, n, fold):
    d = repo / "runs" / stamp / "mae" / f"fold{fold}"
    return any(d.glob("best_*.pth")) or any(d.glob("latest_*.pth"))


def has_monai(stamp, n, fold):
    d = repo / "runs" / stamp / "monai" / f"fold{fold}"
    return any(d.glob("best_*.pth")) or any(d.glob("latest_*.pth"))


def has_nnunet(stamp, n, fold):
    if not stamp:
        return False
    for cand in (work / f"{stamp}_f{fold}", work / stamp):
        if any(cand.glob(f"**/fold_{fold}/checkpoint_*.pth")):
            return True
        if cand.name.endswith(f"_f{fold}") and any(cand.glob("**/checkpoint_*.pth")):
            return True
    return False


def has_dpd(stamp, n, fold):
    return has_nnunet(stamp, n, fold)


def has_seg(stamp, n, fold):
    d = repo / "runs" / stamp / "seganypet" / f"fold{fold}"
    return any(d.glob("*.pth")) or (d / "best.pth").is_file()


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


check = {
    "mae": has_mae,
    "monai": has_monai,
    "nnunet": has_nnunet,
    "dpdnet": has_dpd,
    "seganypet": has_seg,
}

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
                if check[short](stamp, n, fold):
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
  echo "[extra-fold-1gpu] no missing extra folds — mark done"
  {
    echo "done_at=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "status=ok"
    echo "note=all extra folds present"
  } > "${DONE_MARK}"
  exit 0
fi

IFS='|' read -r METH N FOLD STAMP <<< "${JOB}"
echo "[extra-fold-1gpu] job method=${METH} fs${N} fold=${FOLD} stamp=${STAMP} gpu=${GPU}"

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

SPLIT_DIR="${ROOT}/ICLR2026/data/splits_mae_psma_fewshot${N}_9fold"
[[ -f "${SPLIT_DIR}/fold0_nnunet.json" ]] || python3 "${ROOT}/ICLR2026/scripts/export_mae_psma_fewshot50_9fold.py" \
  --n-shot "${N}" --out-dir "${SPLIT_DIR}" --seed 42

STAGE="psma_fs${N}_f258"
case "${METH}" in
  mae) BOARD_KEY="mae_swinunetr" ;;
  monai) BOARD_KEY="monai_swinvit" ;;
  *) BOARD_KEY="${METH}" ;;
esac
export TASK1_FEWSHOT_N="${N}"
export TASK1_PSMA_BOARD_STAGE="${STAGE}"
export TASK1_FEWSHOT_SPLIT_DIR="${SPLIT_DIR}"
export TASK1_NUM_EPOCHS="${TASK1_NUM_EPOCHS:-100}"
export TASK1_TRAIN_ITERS_PER_EPOCH="${TASK1_TRAIN_ITERS_PER_EPOCH:-25}"
export TASK1_VAL_ITERS_PER_EPOCH="${TASK1_VAL_ITERS_PER_EPOCH:-25}"
export TASK1_FS50_VAL_ITERS=25
export TASK1_FS50_VAL_EVERY_N_EPOCHS=20
export TASK1_VAL_EVERY_N_EPOCHS=20
export TASK1_FIXED_BATCH_3D_FULLRES=2
export TASK1_MAE_BATCH_SIZE=2
export TASK1_MAE_NUM_EPOCHS=100

_fail_skip() {
  mkdir -p "$(dirname "${SKIP_FOLDS}")"
  echo "${METH}|${N}|${FOLD}" >> "${SKIP_FOLDS}"
  echo "[extra-fold-1gpu] FAIL skip ${METH} fs${N} f${FOLD} → next fold" >&2
  python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD}" --no-plot \
    --patch-json "{\"updated_note\":\"extra-fold skip ${METH} fs${N} f${FOLD} (failed)\"}" || true
  exit 0
}

_ckpt_ok() {
  REPO="${REPO}" WORK="${WORK}" METH="${METH}" FOLD="${FOLD}" STAMP="${STAMP}" python3 - <<'PY'
import os
from pathlib import Path
repo = Path(os.environ["REPO"])
work = Path(os.environ["WORK"]) / "nnUNet_results"
meth, fold, stamp = os.environ["METH"], os.environ["FOLD"], os.environ["STAMP"]
ok = False
if meth == "mae":
    d = repo / "runs" / stamp / "mae" / f"fold{fold}"
    ok = any(d.glob("best_*.pth")) or any(d.glob("latest_*.pth"))
elif meth == "monai":
    d = repo / "runs" / stamp / "monai" / f"fold{fold}"
    ok = any(d.glob("best_*.pth")) or any(d.glob("latest_*.pth"))
elif meth == "seganypet":
    d = repo / "runs" / stamp / "seganypet" / f"fold{fold}"
    ok = any(d.glob("*.pth")) or (d / "best.pth").is_file()
elif meth in ("nnunet", "dpdnet"):
    for cand in (work / f"{stamp}_f{fold}", work / stamp):
        if any(cand.glob(f"**/fold_{fold}/checkpoint_*.pth")) or (
            cand.name.endswith(f"_f{fold}") and any(cand.glob("**/checkpoint_*.pth"))
        ):
            ok = True
            break
raise SystemExit(0 if ok else 1)
PY
}

_wait_pid_gone() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 0
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 20
  done
}

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"${BOARD_KEY}\":{\"${STAGE}\":{\"status\":\"running\",\"note\":\"9fold extra · ${METH} fs${N} f${FOLD} · GPU ${GPU}\"}}},\"updated_note\":\"gpu-idle extra fold ${METH} fs${N} f${FOLD} GPU ${GPU}\"}" || true

rc=0
set +e
case "${METH}" in
  mae)
    TASK1_MAE_FDG_SEG_CKPT="${TASK1_MAE_FDG_SEG_CKPT:-${REPO}/runs/20260812_072719_iclr2026_mae_fdg_swinbase_gpu013_bs6_tr70_val10_100ep/best_seg_fdg_mae.pth}" \
    TASK1_MAE_FEWSHOT_FOLDS_CSV="${FOLD}" \
    TASK1_MAE_FT_GPU_LIST="${GPU}" \
    TASK1_MAE_SEQ_GPUS="${GPU}" \
    TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" \
      bash "${ROOT}/ICLR2026/run/run_mae_psma_fewshot50_f258_from_fdg_seg_bg.sh"
    ;;
  monai)
    TASK1_MONAI_FDG_SEG_CKPT="${TASK1_MONAI_FDG_SEG_CKPT:-${REPO}/runs/20260816_214921_iclr2026_monai_fdg_swinvit_1gpu_bs6_tr70_val10_100ep/best_seg_fdg_monai.pth}" \
    TASK1_MAE_FEWSHOT_FOLDS_CSV="${FOLD}" \
    TASK1_MAE_FT_GPU_LIST="${GPU}" \
    TASK1_MAE_SEQ_GPUS="${GPU}" \
    TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" \
      bash "${ROOT}/ICLR2026/run/run_monai_psma_fewshot50_f258_from_fdg_seg_bg.sh"
    ;;
  nnunet)
    FOLD_ID="${FOLD}" GPU_ID="${GPU}" PARENT_STAMP="${STAMP}" \
    TASK1_DOCKER_BACKGROUND=0 \
    TASK1_BEST_BY=val_loss TASK1_VAL_LOSS_ONLY=1 \
    TASK1_UDA_FDG_STAMP="${TASK1_UDA_FDG_STAMP:-20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep}" \
    TASK1_UDA_FDG_BEST="${TASK1_UDA_FDG_BEST:-/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth}" \
      bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fewshot50_onefold_bg.sh"
    ;;
  dpdnet)
    FOLD_ID="${FOLD}" GPU_ID="${GPU}" PARENT_STAMP="${STAMP}" \
    TASK1_DPDNET_SKIP_PREPARE=1 TASK1_BEST_BY=val_loss \
    TASK1_DPDNET_NUM_EPOCHS=100 TASK1_DPDNET_TRAIN_ITERS=25 TASK1_DPDNET_VAL_ITERS=25 \
    TASK1_DPDNET_VAL_EVERY=20 TASK1_DPDNET_BATCH_SIZE=2 \
    TASK1_DPDNET_FDG_BEST="${TASK1_DPDNET_FDG_BEST:-/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/20260817_165250_iclr2026_dpdnet_fdg_2gpu_bs3_gbs6_n6_tr70_val0_169ep_gpu01/Dataset239_DpDNet_FDG_2ch/STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth}" \
      bash "${ROOT}/ICLR2026/run/run_dpdnet_psma_fewshot50_onefold_bg.sh"
    _brc=$?
    _dpd_pidf="${WORK}/nnUNet_results/${STAMP}_f${FOLD}/nohup.pid"
    if [[ -f "${_dpd_pidf}" ]]; then
      _wait_pid_gone "$(tr -d '[:space:]' < "${_dpd_pidf}")"
    fi
    _cname="dpdnet_psma_f${FOLD}_${STAMP}_f${FOLD}"
    while docker ps -q -f "name=${_cname}" 2>/dev/null | grep -q .; do
      sleep 20
    done
    (exit "${_brc}")
    ;;
  seganypet)
    TASK1_SEGANY_FOLDS_CSV="${FOLD}" \
    TASK1_SEGANY_GPU_LIST="${GPU}" \
    TASK1_SEGANY_DATA_ROOT="${DATA}/task1_train_workspace/seganypet_fewshot${N}_f258" \
    TASK1_SEGANY_CKPT="${TASK1_SEGANY_CKPT:-${REPO}/runs/20260817_041526_iclr2026_seganypet_fdg_3gpu_bs6_gpu013/seganypet_fdg/best.pth}" \
    TASK1_SEGANY_EPOCHS=100 TASK1_SEGANY_BATCH_SIZE=2 \
    TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" \
      bash "${ROOT}/ICLR2026/run/run_seganypet_fewshot50_f258_bg.sh"
    ;;
  *)
    echo "[error] unknown method ${METH}" >&2
    exit 2
    ;;
esac
rc=$?
set -e
[[ "${rc}" -eq 0 ]] || _fail_skip
_ckpt_ok || _fail_skip

echo "[extra-fold-1gpu] finished ${METH} fs${N} f${FOLD} gpu=${GPU}"
