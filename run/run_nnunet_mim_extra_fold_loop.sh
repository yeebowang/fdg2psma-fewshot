#!/usr/bin/env bash
# Keep training nnUNet MIM missing extra folds on one GPU until none remain.
#   bash ICLR2026/run/run_nnunet_mim_extra_fold_loop.sh --gpu 1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
WORKER="${ROOT}/ICLR2026/run/run_nnunet_mim_extra_fold_onegpu.sh"
GPU=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:?}"; shift 2 ;;
    *) shift ;;
  esac
done
[[ -n "${GPU}" ]] || { echo "[error] need --gpu N" >&2; exit 2; }
chmod +x "${WORKER}"

DONE_MARK="${VIS}/TASK1_NNUNET_MIM_EXTRA_FOLDS_DONE.txt"
echo "[mim-extra-loop] gpu=${GPU} start $(TZ=Asia/Shanghai date '+%F %T')"
while true; do
  # Probe: if worker would find nothing, stop (dry pick via worker once)
  set +e
  bash "${WORKER}" --gpu "${GPU}"
  rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "[mim-extra-loop] gpu=${GPU} rc=${rc} — retry in 60s"
    sleep 60
    continue
  fi
  # Re-check emptiness: try pick without training by seeing if another call immediately says no missing
  # Worker already trained one fold OR reported no missing. Detect no-missing by a second pick attempt
  # that exits quickly when only "no missing folds" is printed — but a second call would train next.
  # Instead: count remaining missing via python.
  remain="$(
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}" \
WORK="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}/task1_train_workspace" \
INFLIGHT="${VIS}/TASK1_NNUNET_MIM_EXTRA_FOLD_INFLIGHT.txt" \
python3 - <<'PY'
import json, os
from pathlib import Path
board = json.loads(Path(os.environ["BOARD"]).read_text(encoding="utf-8"))
work = Path(os.environ["WORK"]) / "nnUNet_results"
methods = board.get("methods") or {}
extra = [0, 1, 3, 4, 6, 7]
n_miss = 0
for n in (50, 10, 5):
    st = ((methods.get("nnunet_mim") or {}).get(f"psma_fs{n}_f258") or {}).get("stamp") or ""
    if not st:
        continue
    for fold in extra:
        ok = False
        for cand in (work / f"{st}_f{fold}", work / st):
            if any(cand.glob(f"**/fold_{fold}/checkpoint_*.pth")) or (
                cand.name.endswith(f"_f{fold}") and any(cand.glob("**/checkpoint_*.pth"))
            ):
                ok = True
                break
        if not ok:
            n_miss += 1
print(n_miss)
PY
)"
  echo "[mim-extra-loop] gpu=${GPU} remaining_missing=${remain}"
  if [[ "${remain}" -eq 0 ]]; then
    {
      echo "done_at=$(TZ=Asia/Shanghai date '+%F %T %Z')"
      echo "status=ok"
      echo "by=gpu${GPU}"
    } > "${DONE_MARK}"
    echo "[mim-extra-loop] gpu=${GPU} all done $(TZ=Asia/Shanghai date '+%F %T')"
    break
  fi
done
