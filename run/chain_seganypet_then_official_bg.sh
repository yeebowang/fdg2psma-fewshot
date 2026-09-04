#!/usr/bin/env bash
# Wait for current SegAnyPET light fewshot (parallel 1GPU/fold) to finish,
# then launch official-hparams run (sequential folds, 3-GPU each).
set -euo pipefail

CTRL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WAIT_STAMP="${1:-20260815_125559_iclr2026_seganypet_fs50_f258_gpu013}"
RUN="${CTRL}/ICLR2026/3D-MAE-PET-CT/runs/${WAIT_STAMP}"
LOG=/tmp/seganypet_chain_official.log

echo "[chain] waiting for ${WAIT_STAMP} …" | tee -a "${LOG}"
while true; do
  if [[ -f "${RUN}/aggregate_val_dice_f258.json" ]]; then
    echo "[chain] found aggregate → start official" | tee -a "${LOG}"
    break
  fi
  # pipeline + containers gone for a while
  if ! pgrep -f run_seganypet_fewshot50_f258_bg.sh >/dev/null \
     && ! docker ps --format '{{.Names}}' | grep -q "seganypet_fs50_.*${WAIT_STAMP}"; then
    sleep 120
    if [[ -f "${RUN}/aggregate_val_dice_f258.json" ]]; then
      break
    fi
    if ! pgrep -f run_seganypet_fewshot50_f258_bg.sh >/dev/null \
       && ! docker ps --format '{{.Names}}' | grep -q "seganypet_fs50_.*${WAIT_STAMP}"; then
      echo "[chain] train processes gone but no aggregate — check logs; still starting official if folds look done" | tee -a "${LOG}"
      # require all 3 folds reached 100ep
      ok=0
      for f in 2 5 8; do
        m="${RUN}/seganypet/fold${f}/metrics.jsonl"
        [[ -f "${m}" ]] || continue
        ep=$(tail -1 "${m}" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read()).get("epoch",0))' 2>/dev/null || echo 0)
        [[ "${ep}" -ge 100 ]] && ok=$((ok + 1))
      done
      if [[ "${ok}" -ge 3 ]]; then
        break
      fi
      echo "[chain] incomplete (ok_folds=${ok}); exit 1" | tee -a "${LOG}"
      exit 1
    fi
  fi
  sleep 60
done

# disarm previous stamp arm, then fresh official stamp
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
unset TASK1_NNUNET_RESULTS_STAMP_NAME
export TASK1_BASE=/media/ybwang/data1/PSMA-DATA
export TASK1_SEGANY_CKPT="${CTRL}/ICLR2026/3D-MAE-PET-CT/weights/seganypet/seganypet_lesion.pth"
export TASK1_SEGANY_VAL_MAX_CASES=15

# progress watcher for official (stamp filled after start)
nohup env -u TASK1_NNUNET_RESULTS_STAMP_NAME \
  TASK1_BASE=/media/ybwang/data1/PSMA-DATA \
  TASK1_SEGANY_CKPT="${TASK1_SEGANY_CKPT}" \
  TASK1_SEGANY_VAL_MAX_CASES=15 \
  bash "${CTRL}/ICLR2026/run/run_seganypet_fewshot50_f258_official_bg.sh" \
  >>"${LOG}" 2>&1 &
echo $! >/tmp/seganypet_official_pipe.pid
echo "[chain] official pipeline pid=$(cat /tmp/seganypet_official_pipe.pid)" | tee -a "${LOG}"

# attach progress watcher once stamp known
for i in $(seq 1 60); do
  STAMP=$(grep -oE 'STAMP=[0-9]{8}_[0-9]{6}_iclr2026_seganypet_official[^ ]*' "${LOG}" | tail -1 | cut -d= -f2 || true)
  [[ -n "${STAMP}" ]] && break
  sleep 5
done
if [[ -n "${STAMP:-}" ]]; then
  cat >/tmp/watch_seganypet_official_progress.sh <<EOF
#!/bin/bash
STAMP=${STAMP}
CTRL=${CTRL}
RUN=\${CTRL}/ICLR2026/3D-MAE-PET-CT/runs/\${STAMP}
PNG=\${CTRL}/ICLR2026/vis/progress_iclr2026_seganypet_official_f258_\${STAMP}.png
while true; do
  docker run --rm -v \${CTRL}:\${CTRL} iclr2026_3dmae_petct:cu118 \\
    python3 \${CTRL}/ICLR2026/scripts/seganypet_f258_progress_plot.py \\
      --run-root \${RUN} --out-png \${PNG} --folds 2,5,8 --ft-epochs 200 \\
    >>/tmp/seganypet_official_progress.log 2>&1 || true
  [[ -f \${RUN}/aggregate_val_dice_f258.json ]] && exit 0
  if ! pgrep -f run_seganypet_fewshot50_f258_official_bg.sh >/dev/null \\
     && ! docker ps --format '{{.Names}}' | grep -q seganypet_official; then
    sleep 90
    if ! pgrep -f run_seganypet_fewshot50_f258_official_bg.sh >/dev/null \\
       && ! docker ps --format '{{.Names}}' | grep -q seganypet_official; then exit 0; fi
  fi
  sleep 45
done
EOF
  chmod +x /tmp/watch_seganypet_official_progress.sh
  nohup /tmp/watch_seganypet_official_progress.sh >/tmp/seganypet_official_progress_outer.log 2>&1 &
  echo "[chain] official STAMP=${STAMP} watcher=$!" | tee -a "${LOG}"
fi

wait "$(cat /tmp/seganypet_official_pipe.pid)"
echo "[chain] official finished rc=$?" | tee -a "${LOG}"
