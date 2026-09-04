#!/usr/bin/env bash
# Exclusive 3-GPU algo queue (one method at a time · GPUs 0,1,3):
#   MONAI → SegAnyPET → DpDNet (FDG+PSMA) →
#   nnUNet aligned: FDG 169ep → PSMA tr25/val25e20/100ep → TEST20
#
# Assumes MONAI chain (wait_monai_fdg_then_fs50_test_gpu013) already running or DONE mark present.
set -euo pipefail

CTRL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"
POLL_SEC="${TASK1_CHAIN_POLL_SEC:-60}"

MONAI_DONE="${TASK1_MONAI_DONE_MARK:-${LOG_DIR}/TASK1_MONAI_PIPELINE_DONE.txt}"
SEGANY_DONE="${TASK1_SEGANY_DONE_MARK:-${LOG_DIR}/TASK1_SEGANY_PIPELINE_DONE.txt}"
DPDNET_DONE="${TASK1_DPDNET_DONE_MARK:-${LOG_DIR}/TASK1_DPDNET_PIPELINE_DONE.txt}"
NNUNET_DONE="${TASK1_NNUNET_DONE_MARK:-${LOG_DIR}/TASK1_NNUNET_PSMA_PIPELINE_DONE.txt}"
FROM="${TASK1_ALGO_QUEUE_FROM:-after_monai}"  # after_monai|seganypet|dpdnet|nnunet

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"

_log() { echo "[algo-queue] $*"; }

_wait_ok_mark() {
  local mark="$1" label="$2"
  while [[ ! -f "${mark}" ]] || ! grep -q 'status=ok' "${mark}"; do
    _log "waiting ${label}… $(TZ=Asia/Shanghai date +%H:%M:%S)"
    # also accept: monai chain still alive (heartbeat)
    sleep "${POLL_SEC}"
  done
  _log "${label} done: $(cat "${mark}")"
}

_board_q() {
  local note="$1" q="$2"
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" \
    --patch-json "{\"updated_note\":\"${note}\",\"queue\":${q}}" || true
}

_log "FROM=${FROM} queue=MONAI→SegAnyPET→DpDNet→nnUNetPSMA+TEST · exclusive 3GPU"

case "${FROM}" in
  after_monai)
    _board_q "queue: wait MONAI then SegAnyPET→DpDNet→nnUNet" \
      '["monai(running)","seganypet","dpdnet","nnunet.psma+test"]'
    _wait_ok_mark "${MONAI_DONE}" "MONAI FDG+FS+TEST"
    ;&
  seganypet)
    _board_q "queue: SegAnyPET next" '["seganypet","dpdnet","nnunet.psma+test"]'
    # clear stale seganypet DONE so we re-run
    rm -f "${SEGANY_DONE}" || true
    bash "${CTRL}/ICLR2026/run/run_seganypet_aligned_full_gpu013_bg.sh"
    ;&
  dpdnet)
    _board_q "queue: DpDNet FDG 3gpu" '["dpdnet","nnunet.psma+test"]'
    rm -f "${DPDNET_DONE}" || true
    export TASK1_DPDNET_GPU_POOL=0,1,3
    export TASK1_DPDNET_FORCE_TIER=3
    export TASK1_DPDNET_FORCE_GPUS=0,1,3
    export TASK1_DPDNET_BATCH_SIZE=6
    export TASK1_DPDNET_NUM_EPOCHS="${TASK1_DPDNET_NUM_EPOCHS:-100}"
    export TASK1_DPDNET_SKIP_PREPARE=1
    bash "${CTRL}/ICLR2026/run/run_dpdnet_fdg_bs_ladder_bg.sh"
    # PSMA for DpDNet not wired yet — mark pending and continue
    python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
      --board "${BOARD_JSON}" \
      --patch-json '{"methods":{"dpdnet":{"psma_fs50_f258":{"status":"pending","note":"FDG done on 3gpu; PSMA/TEST script not ready — skipped in this queue"}}},"updated_note":"DpDNet FDG done; PSMA pending code"}' || true
    {
      echo "done_at=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
      echo "status=ok"
      echo "note=fdg_only_psma_pending"
    } >"${DPDNET_DONE}"
    ;&
  nnunet)
    # 对齐 DpDNet：FDG 169ep → PSMA tr25/val25e20/100ep → TEST20
    _board_q "queue: nnUNet aligned FDG169→PSMA→TEST" \
      '["nnunet.fdg 169ep","nnunet.psma f258","nnunet.test20"]'
    rm -f "${NNUNET_DONE}" || true
    export TASK1_NNUNET_FDG_EPOCHS=169
    export TASK1_NNUNET_PSMA_EPOCHS=100
    export TASK1_NNUNET_PSMA_TRAIN_ITERS=25
    export TASK1_NNUNET_PSMA_VAL_ITERS=25
    export TASK1_NNUNET_PSMA_VAL_EVERY=20
    export TASK1_FIXED_BATCH_3D_FULLRES=2
    export TASK1_BEST_BY=val_loss
    export TASK1_VAL_LOSS_ONLY=1
    bash "${CTRL}/ICLR2026/run/run_nnunet_aligned_fdg169_psma_f258_bg.sh"
    {
      echo "done_at=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
      echo "status=ok"
      echo "note=aligned_fdg169_psma_tr25_val25e20_100ep"
    } >"${NNUNET_DONE}"
    _board_q "ALL ALGO QUEUE DONE" '[]'
    ;;
  *)
    echo "[error] unknown TASK1_ALGO_QUEUE_FROM=${FROM}" >&2
    exit 2
    ;;
esac

_log "ALL DONE"
