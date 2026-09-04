#!/usr/bin/env bash
# Worker:
#   A) CPU rescore immediately (no wait for GPU/fc70) — reuse nifti preds
#   B) GPU re-eval only after fc70 frees GPUs — MAE/MONAI (no nifti) / nnUNet if incomplete
#
# Invoked by run_rescore_dice_fp_fn_after_queues_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
POLL_SEC="${TASK1_CHAIN_POLL_SEC:-60}"
PID_FILE="${VIS}/rescore_metrics_queue.pid"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
FDG_EVAL="${WORK}/fdg_test20_eval"

echo $$ > "${PID_FILE}"
echo "[rescore-queue] $(date '+%F %T') start pid=$$ (CPU now · GPU waits)"

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" \
  --patch-json '{"updated_note":"rescore · RUNNING (CPU) · GPU waits after"}' || true

echo "[rescore-queue] === CPU phase (no wait) ==="
python3 "${ROOT}/ICLR2026/scripts/rescore_board_dice_fp_fn.py" --board "${BOARD}" || true
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" \
  --patch-json '{"updated_note":"CPU rescore DONE · GPU pending"}' || true
echo "[rescore-queue] CPU phase DONE $(date '+%F %T')"

_busy_fc70() {
  pgrep -af 'run_aligned_psma_fc70_pipeline_bg.sh|run_mae_psma_fc70|run_nnunet_psma_fc70|run_monai_psma_fc70|run_dpdnet_psma_fc70' 2>/dev/null \
    | grep -Ev 'rescore_dice_fp_fn|queue_keeper|run_psma_fc70_after_main_queue_bg' | grep -q .
}

_busy_fdg_eval() {
  pgrep -af 'run_eval_fdg_shared_test20_bg.sh|run_eval_fdg_test20_bg.sh|run_fdg_eval_after_fc70_queue' 2>/dev/null \
    | grep -Ev 'rescore_dice_fp_fn|queue_keeper' | grep -q .
}

_need_gpu_mae_monai() {
  VIS="${VIS}" python3 - <<'PY'
import json, os
from pathlib import Path
agg = Path(os.environ["VIS"]) / "fdg_test20"
need = []
for m, key in (("mae", "aggregate_mae_swinunetr.json"), ("monai", "aggregate_monai_swinvit.json")):
    p = agg / key
    ok = False
    if p.is_file():
        d = json.loads(p.read_text())
        fp, fn = d.get("fp_rate", d.get("mean_fp")), d.get("fn_rate", d.get("mean_fn"))
        ok = isinstance(fp, (int, float)) and fp == fp and isinstance(fn, (int, float)) and fn == fn
    if not ok:
        need.append(m)
print(",".join(need))
PY
}

_need_nnunet_gpu() {
  VIS="${VIS}" FDG_EVAL="${FDG_EVAL}" python3 - <<'PY'
import json, os
from pathlib import Path
agg = Path(os.environ["VIS"]) / "fdg_test20/aggregate_nnunet.json"
pred = Path(os.environ["FDG_EVAL"]) / "nnunet/predict/pred"
n = len(list(pred.glob("*.nii.gz"))) if pred.is_dir() else 0
fp = fn = None
if agg.is_file():
    d = json.loads(agg.read_text())
    fp, fn = d.get("fp_rate", d.get("mean_fp")), d.get("fn_rate", d.get("mean_fn"))
has = isinstance(fp, (int, float)) and fp == fp and isinstance(fn, (int, float)) and fn == fn
print("1" if (not has and n < 180) else "0")
PY
}

need="$(_need_gpu_mae_monai)"
need_nn="$(_need_nnunet_gpu)"
if [[ -z "${need}" && "${need_nn}" != "1" ]]; then
  echo "[rescore-queue] no GPU work needed — ALL DONE $(date '+%F %T')"
  python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" \
    --patch-json '{"updated_note":"rescore Dice/FP/FN DONE (CPU only; no GPU needed)"}' || true
  rm -f "${PID_FILE}"
  exit 0
fi

echo "[rescore-queue] === GPU phase (wait for free GPU / fc70) need=${need:-none} nnunet=${need_nn} ==="
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" \
  --patch-json '{"updated_note":"rescore · waiting GPU · then RUNNING (GPU)"}' || true
while _busy_fc70; do
  echo "[rescore-queue] waiting GPU/fc70… $(TZ=Asia/Shanghai date +%H:%M:%S)"
  sleep "${POLL_SEC}"
done
while _busy_fdg_eval; do
  echo "[rescore-queue] waiting other fdg eval… $(TZ=Asia/Shanghai date +%H:%M:%S)"
  sleep "${POLL_SEC}"
done

if [[ -n "${need}" ]]; then
  export TASK1_TEST_SKIP_DONE=0
  IFS=',' read -r -a arr <<< "${need}"
  for m in "${arr[@]}"; do
    [[ -z "${m}" ]] && continue
    gpus="${CUDA_VISIBLE_DEVICES:-${TASK1_GPUS:-0,1,3}}"
    echo "[rescore-queue] GPU re-eval METHOD=${m} gpus=${gpus}"
    python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" \
      --patch-json "{\"updated_note\":\"rescore · RUNNING (GPU ${gpus}) · ${m}\"}" || true
    METHOD="${m}" bash "${ROOT}/ICLR2026/run/run_eval_fdg_test20_bg.sh" || echo "[rescore-queue] warn ${m} failed"
  done
else
  echo "[rescore-queue] skip MAE/MONAI GPU"
fi

if [[ "${need_nn}" == "1" ]]; then
  echo "[rescore-queue] nnUNet FDG incomplete → GPU once"
  export TASK1_TEST_SKIP_DONE=0
  METHOD=nnunet bash "${ROOT}/ICLR2026/run/run_eval_fdg_test20_bg.sh" || echo "[rescore-queue] warn nnunet failed"
else
  echo "[rescore-queue] skip nnUNet GPU"
fi

python3 "${ROOT}/ICLR2026/scripts/rescore_board_dice_fp_fn.py" --board "${BOARD}" || true
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" \
  --patch-json '{"updated_note":"rescore Dice/FP/FN DONE (CPU+GPU)"}' || true

echo "[rescore-queue] ALL DONE $(date '+%F %T')"
rm -f "${PID_FILE}"
