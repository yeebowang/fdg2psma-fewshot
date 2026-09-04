#!/usr/bin/env bash
# Wait for SegAnyPET official bs3×1GPU f258 to finish, then run Baseline1 FDG-only
# zero-shot eval on shared PSMA val, reported for full fewshot50 folds 0..8.
set -euo pipefail

CTRL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WAIT_STAMP="${1:-20260815_222549_iclr2026_seganypet_official_bs3_1gpu_f258_gpu013}"
RUN="${CTRL}/ICLR2026/3D-MAE-PET-CT/runs/${WAIT_STAMP}"
LOG="${CTRL}/ICLR2026/vis/chain_seganypet_bs3_then_baseline1_eval_${WAIT_STAMP}.log"
TARGET_EP="${CHAIN_TARGET_EP:-200}"

echo "[chain] waiting for SegAnyPET ${WAIT_STAMP} (ep>=${TARGET_EP} × f2/5/8) …" | tee -a "${LOG}"
echo "[chain] next: Baseline1 FDG → PSMA val 9fold (f0..f8)" | tee -a "${LOG}"

_done_folds() {
  local ok=0 f m ep
  for f in 2 5 8; do
    m="${RUN}/seganypet/fold${f}/metrics.jsonl"
    [[ -f "${m}" ]] || continue
    ep=$(tail -1 "${m}" | python3 -c 'import sys,json; print(int(json.loads(sys.stdin.read()).get("epoch",0)))' 2>/dev/null || echo 0)
    [[ "${ep}" -ge "${TARGET_EP}" ]] && ok=$((ok + 1))
  done
  echo "${ok}"
}

while true; do
  if [[ -f "${RUN}/aggregate_val_dice_f258.json" ]]; then
    echo "[chain] found aggregate" | tee -a "${LOG}"
    break
  fi
  ok="$(_done_folds)"
  if [[ "${ok}" -ge 3 ]]; then
    # wait until docker/train scripts also exit
    if ! pgrep -f "run_seganypet_fewshot50_f258_bg.sh" >/dev/null \
       && ! docker ps --format '{{.Names}}' | grep -q "seganypet_fs50_.*${WAIT_STAMP}"; then
      echo "[chain] all folds ≥${TARGET_EP} and processes gone" | tee -a "${LOG}"
      break
    fi
    echo "[chain] folds done (${ok}/3) but train still running; wait…" | tee -a "${LOG}"
  else
    echo "[chain] progress folds_done=${ok}/3 $(date '+%H:%M:%S')" | tee -a "${LOG}"
  fi
  sleep 60
done

# optional: reeval click-val with fixed infer (does not block baseline1 if it fails)
if [[ ! -f "${RUN}/aggregate_val_dice_f258_reeval.json" ]]; then
  echo "[chain] launching seganypet reeval (best-effort) …" | tee -a "${LOG}"
  nohup docker run --rm --name "seganypet_reeval_${WAIT_STAMP}" \
    --gpus '"device=0"' \
    -e PYTHONPATH="${CTRL}/ICLR2026/third_party/seganypet_pip:${CTRL}/ICLR2026/third_party/SegAnyPET/code:${CTRL}/ICLR2026/scripts" \
    -v "${CTRL}:${CTRL}" -v /media/ybwang/data1/PSMA-DATA:/media/ybwang/data1/PSMA-DATA \
    -w "${CTRL}/ICLR2026/third_party/SegAnyPET/code" \
    --shm-size=8g \
    iclr2026_3dmae_petct:cu118 \
    python3 "${CTRL}/ICLR2026/scripts/seganypet_reeval_f258.py" \
      --run-root "${RUN}" --folds 2,5,8 --num-clicks 5 \
    >>"${LOG}" 2>&1 &
  echo $! >"/tmp/seganypet_reeval_${WAIT_STAMP}.pid"
fi

echo "[chain] start Baseline1 FDG → PSMA val 9fold (f0..f8) eval" | tee -a "${LOG}"
unset TASK1_NNUNET_RESULTS_STAMP_NAME
nohup env -u TASK1_NNUNET_RESULTS_STAMP_NAME \
  TASK1_BASE=/media/ybwang/data1/PSMA-DATA \
  TASK1_EVAL_FOLDS=0,1,2,3,4,5,6,7,8 \
  bash "${CTRL}/ICLR2026/run/run_baseline1_fdg_eval_psma_f258_bg.sh" \
  >>"${LOG}" 2>&1 &
echo $! >"/tmp/baseline1_eval_9fold_pipe.pid"
echo "[chain] baseline1 eval pid=$(cat /tmp/baseline1_eval_9fold_pipe.pid) log=${LOG}" | tee -a "${LOG}"
