#!/usr/bin/env bash
# Continuation of aligned FDG→PSMA fs50 pipeline after an in-flight MAE fewshot.
# Prefer: run_aligned_pipeline_stages_bg.sh (FROM_STAGE=...) + watchdog.
# This script kept for MAE-wait entry; board patches use Python helper (no fragile bash JSON).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"
BOARD_PNG="${TASK1_ALIGN_BOARD_PNG:-${LOG_DIR}/progress_iclr2026_aligned_fdg_fs50_f258_board.png}"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"
export TASK1_BASE="${DATA}"

MAE_STAMP="${TASK1_MAE_WAIT_STAMP:-20260816_161859_iclr2026_mae_psma_fs50_from_fdg_seg_f258_gpu013}"
PIPE_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_aligned_continue_nnunet_first"
PIPE_LOG="${LOG_DIR}/nohup_aligned_continue_${PIPE_STAMP}.log"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[aligned-cont] PIPE=${PIPE_STAMP} wait_mae=${MAE_STAMP}"
echo "[aligned-cont] after MAE TEST20 → hand off to run_aligned_pipeline_stages_bg.sh"

_board() {
  python3 "${CTRL}/ICLR2026/scripts/aligned_board_patch.py" --board "${BOARD_JSON}" "$@" || true
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" --png "${BOARD_PNG}" || true
}

_board --queue "mae_swinunetr.psma_fs50_f258 running,nnunet PRIORITY,monai,seganypet" \
  --set-stage nnunet.psma_fs50_f258 --status pending \
  --extra '{"bs":2,"bs_note":"per-GPU re-run","stamp":"","mean":null,"fold_dice":{},"note":"queued PRIORITY"}' \
  --updated-note "waiting MAE then nnUNet bs=2"

pgrep -f 'iclr2026_aligned_fdg_fs50_board.py --watch' >/dev/null 2>&1 || \
  nohup python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --watch 60 \
    >"${LOG_DIR}/nohup_aligned_board_watch_cont.log" 2>&1 &

# start watchdog early
pgrep -f 'run_aligned_pipeline_watchdog_bg.sh' >/dev/null 2>&1 || \
  nohup bash "${CTRL}/ICLR2026/run/run_aligned_pipeline_watchdog_bg.sh" >/dev/null 2>&1 &

echo "[aligned-cont] waiting MAE containers / ckpts for ${MAE_STAMP}"
while true; do
  n_live=$(docker ps --format '{{.Names}}' | grep -c "mae_fs50_fdgseg_.*_${MAE_STAMP}" || true)
  if [[ "${n_live}" -eq 0 ]]; then
    ok=0
    for f in 2 5 8; do
      [[ -f "${REPO}/runs/${MAE_STAMP}/mae/fold${f}/latest_seg_psma_fs50_fdgseg_f${f}.pth" ]] && ok=$((ok+1))
    done
    if [[ "${ok}" -eq 3 ]]; then
      echo "[aligned-cont] MAE folds ready"
      break
    fi
  fi
  echo "[aligned-cont] MAE still running containers=${n_live} $(date +%H:%M:%S)"
  sleep 60
done

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" --ingest-mae-stamp "${MAE_STAMP}" || true

echo "[aligned-cont] === MAE TEST20 eval ==="
_board --set-stage mae_swinunetr.psma_fs50_f258 --status running \
  --stamp "${MAE_STAMP}" --note "TEST20 eval" \
  --extra '{"phase":"TEST20","epoch":100,"total_epochs":100}' \
  --queue "mae TEST20,nnunet PRIORITY,monai,seganypet" \
  --updated-note "MAE TEST20 eval"
METHOD=mae STAMP="${MAE_STAMP}" \
  bash "${CTRL}/ICLR2026/run/run_eval_psma_test20_f258_bg.sh" \
  || { echo "[aligned-cont][FAIL] MAE TEST20"; exit 1; }

_require_mae_agg="${REPO}/runs/${MAE_STAMP}/psma_test20_eval/aggregate_test20_f258.json"
[[ -f "${_require_mae_agg}" ]] || { echo "[aligned-cont][FAIL] missing ${_require_mae_agg}"; exit 1; }

echo "[aligned-cont] hand off → stages FROM_STAGE=nnunet"
# Replace self with stages runner (fresh process; no mid-file edit risk)
exec env FROM_STAGE=nnunet bash "${CTRL}/ICLR2026/run/run_aligned_pipeline_stages_bg.sh"
