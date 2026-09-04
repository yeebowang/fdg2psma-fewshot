#!/usr/bin/env bash
# Loop: fill nnUNet MIM missing TEST20 extra folds on one GPU.
#   bash ICLR2026/run/run_nnunet_mim_extra_fold_test20_loop.sh --gpu 1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
WORKER="${ROOT}/ICLR2026/run/run_nnunet_mim_extra_fold_test20_onegpu.sh"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
DONE_MARK="${VIS}/TASK1_NNUNET_MIM_EXTRA_TEST20_DONE.txt"

GPU=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:?}"; shift 2 ;;
    *) shift ;;
  esac
done
[[ -n "${GPU}" ]] || { echo "[error] need --gpu N" >&2; exit 2; }
chmod +x "${WORKER}"

_remain() {
BOARD="${BOARD}" WORK="${WORK}" python3 - <<'PY'
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
        ckpt = False
        for cand in (work / f"{st}_f{fold}", work / st):
            if any(cand.glob(f"**/fold_{fold}/checkpoint_*.pth")) or (
                cand.name.endswith(f"_f{fold}") and any(cand.glob("**/checkpoint_*.pth"))
            ):
                ckpt = True
                break
        scored = (work / st / "psma_test20_eval" / f"fold{fold}" / "score_detail.json").is_file()
        if ckpt and not scored:
            n_miss += 1
print(n_miss)
PY
}

echo "[mim-test20-loop] gpu=${GPU} start $(TZ=Asia/Shanghai date '+%F %T')"
while true; do
  set +e
  bash "${WORKER}" --gpu "${GPU}"
  rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "[mim-test20-loop] gpu=${GPU} rc=${rc} — retry in 60s"
    sleep 60
    continue
  fi
  remain="$(_remain)"
  echo "[mim-test20-loop] gpu=${GPU} remaining_missing=${remain}"
  if [[ "${remain}" -eq 0 ]]; then
    {
      echo "done_at=$(TZ=Asia/Shanghai date '+%F %T %Z')"
      echo "status=ok"
      echo "by=gpu${GPU}"
    } > "${DONE_MARK}"
    echo "[mim-test20-loop] gpu=${GPU} all done $(TZ=Asia/Shanghai date '+%F %T')"
    break
  fi
done
