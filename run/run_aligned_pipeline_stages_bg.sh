#!/usr/bin/env bash
# Aligned pipeline stages with filesystem state machine (safe to re-enter).
#   FROM_STAGE=nnunet|monai|seganypet  bash ICLR2026/run/run_aligned_pipeline_stages_bg.sh
#
# Writes heartbeat + state under ICLR2026/vis/aligned_pipeline_state.json
# so the watchdog can detect stalls and resume.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"
BOARD_PNG="${TASK1_ALIGN_BOARD_PNG:-${LOG_DIR}/progress_iclr2026_aligned_fdg_fs50_f258_board.png}"
STATE_JSON="${LOG_DIR}/aligned_pipeline_state.json"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"
export TASK1_BASE="${DATA}"

FROM_STAGE="${FROM_STAGE:-nnunet}"   # nnunet | monai | seganypet
MAE_STAMP="${TASK1_MAE_WAIT_STAMP:-20260816_161859_iclr2026_mae_psma_fs50_from_fdg_seg_f258_gpu013}"
PIPE_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_aligned_stages_${FROM_STAGE}"
PIPE_LOG="${LOG_DIR}/nohup_aligned_stages_${PIPE_STAMP}.log"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[aligned-stages] PIPE=${PIPE_STAMP} FROM=${FROM_STAGE}"

_hb() {
  # heartbeat for watchdog
  STAGE_NAME="$1" STAGE_DETAIL="$2" STATE_JSON="${STATE_JSON}" PIPE_STAMP="${PIPE_STAMP}" PIPE_LOG="${PIPE_LOG}" python3 - <<'PY'
import json, time, os
from pathlib import Path
p = Path(os.environ["STATE_JSON"])
d = {}
if p.is_file():
    try:
        d = json.loads(p.read_text())
    except Exception:
        d = {}
d.update({
    "pipe_stamp": os.environ.get("PIPE_STAMP", ""),
    "pid": os.getppid(),
    "stage": os.environ.get("STAGE_NAME", ""),
    "detail": os.environ.get("STAGE_DETAIL", ""),
    "heartbeat_unix": time.time(),
    "log": os.environ.get("PIPE_LOG", ""),
})
p.write_text(json.dumps(d, indent=2) + "\n")
print("[hb]", d["stage"], d["detail"])
PY
}

_board() {
  python3 "${CTRL}/ICLR2026/scripts/aligned_board_patch.py" --board "${BOARD_JSON}" "$@" || true
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" --png "${BOARD_PNG}" || true
}

_fail() {
  echo "[aligned-stages][FAIL] $*" >&2
  _hb "FAILED" "$*"
  _board --updated-note "PIPELINE FAIL: $*" --queue "FAILED"
  exit 1
}

_require_file() {
  local f="$1" msg="$2"
  [[ -f "${f}" ]] || _fail "${msg}: missing ${f}"
}

# ensure board watcher
pgrep -f 'iclr2026_aligned_fdg_fs50_board.py --watch' >/dev/null 2>&1 || \
  nohup python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --watch 60 \
    >"${LOG_DIR}/nohup_aligned_board_watch_cont.log" 2>&1 &

run_nnunet() {
  local NN_STAMP="${TASK1_NNUNET_FS_STAMP:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_nnunet_psma_fs50_f258_1gpu_bs2_tr70_val70e20_300ep_gpu013}"
  _hb "nnunet" "${NN_STAMP}"
  _board --set-stage nnunet.psma_fs50_f258 --status running --stamp "${NN_STAMP}" \
    --note "PRIORITY bs=2 val70e20" \
    --extra '{"bs":2,"bs_note":"per-GPU","total_epochs":300,"online_val":"VAL70 every20","fold_dice":{},"mean":null}' \
    --queue "nnunet.psma_fs50_f258 running,monai,seganypet" \
    --updated-note "nnUNet PSMA bs=2 running"

  TASK1_FIXED_BATCH_3D_FULLRES=2 \
  TASK1_FS50_VAL_ITERS=70 \
  TASK1_FS50_VAL_EVERY_N_EPOCHS=20 \
  TASK1_NUM_EPOCHS=300 \
  TASK1_TRAIN_ITERS_PER_EPOCH=70 \
  TASK1_NNUNET_RESULTS_STAMP_NAME="${NN_STAMP}" \
    bash "${CTRL}/ICLR2026/run/run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh" \
    || _fail "nnUNet fewshot script failed"

  local NN_AGG="${LOG_DIR}/aggregate_nnunet_psma_fs50_f258_${NN_STAMP}.json"
  _require_file "${NN_AGG}" "nnUNet TEST20 aggregate"
  python3 - <<PY
import json
from pathlib import Path
board_p = Path("${BOARD_JSON}")
board = json.loads(board_p.read_text())
agg = json.loads(Path("${NN_AGG}").read_text())
folds = {k: v.get("best_val_dice") for k, v in agg.get("folds", {}).items()}
st = board["methods"]["nnunet"]["psma_fs50_f258"]
st.update({
    "status": "done", "stamp": "${NN_STAMP}", "bs": 2, "bs_note": "per-GPU",
    "fold_dice": folds, "mean": agg.get("fold_mean"),
    "note": "FDG=6/PSMA=2 · val70e20", "phase": None, "eta": None, "eta_sec": None,
})
board["updated_note"] = f"nnUNet TEST20 mean={agg.get('fold_mean')}"
board_p.write_text(json.dumps(board, indent=2) + "\n")
print("nnunet ingested", st.get("mean"))
PY
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD_JSON}" || true
  _hb "nnunet_done" "${NN_STAMP}"
  export _LAST_NN_STAMP="${NN_STAMP}"
}

run_monai() {
  local MONAI_FDG_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_monai_fdg_swinvit_gpu013_bs6_tr70_val10_100ep"
  _hb "monai_fdg" "${MONAI_FDG_STAMP}"
  _board --set-stage monai_swinvit.fdg_pretrain --status running --stamp "${MONAI_FDG_STAMP}" \
    --extra '{"bs":6,"bs_note":"global 2x3GPU"}' \
    --queue "monai_swinvit.fdg_pretrain running,monai fs,seganypet" \
    --updated-note "MONAI FDG running"

  TASK1_NNUNET_RESULTS_STAMP_NAME="${MONAI_FDG_STAMP}" \
    bash "${CTRL}/ICLR2026/run/run_monai_fdg_swinbase_finetune_100ep_bg.sh" \
    || _fail "MONAI FDG failed"

  local MONAI_BEST="${REPO}/runs/${MONAI_FDG_STAMP}/best_seg_fdg_monai.pth"
  [[ -f "${MONAI_BEST}" ]] || MONAI_BEST="${REPO}/runs/${MONAI_FDG_STAMP}/latest_seg_fdg_monai.pth"
  _require_file "${MONAI_BEST}" "MONAI FDG ckpt"

  _board --set-stage monai_swinvit.fdg_pretrain --status done --stamp "${MONAI_FDG_STAMP}" \
    --extra "{\"best_ckpt\":\"${MONAI_BEST}\"}" --updated-note "MONAI FDG done"

  local MONAI_FS_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_monai_psma_fs50_from_fdg_seg_f258_gpu013"
  _hb "monai_fs" "${MONAI_FS_STAMP}"
  _board --set-stage monai_swinvit.psma_fs50_f258 --status running --stamp "${MONAI_FS_STAMP}" \
    --extra "{\"bs\":2,\"foundation\":\"${MONAI_BEST}\"}" \
    --queue "monai_swinvit.psma_fs50_f258 running,seganypet" \
    --updated-note "MONAI fewshot running"

  TASK1_MONAI_FDG_SEG_CKPT="${MONAI_BEST}" \
  TASK1_MAE_BATCH_SIZE=2 \
  TASK1_NNUNET_RESULTS_STAMP_NAME="${MONAI_FS_STAMP}" \
    bash "${CTRL}/ICLR2026/run/run_monai_psma_fewshot50_f258_from_fdg_seg_bg.sh" \
    || _fail "MONAI fewshot failed"

  METHOD=monai STAMP="${MONAI_FS_STAMP}" \
    bash "${CTRL}/ICLR2026/run/run_eval_psma_test20_f258_bg.sh" \
    || _fail "MONAI TEST20 failed"
  _hb "monai_done" "${MONAI_FS_STAMP}"
}

run_seganypet() {
  local SEG_FDG_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_fdg_pretrain_gpu013"
  _hb "seganypet_fdg" "${SEG_FDG_STAMP}"
  _board --set-stage seganypet.fdg_pretrain --status running --stamp "${SEG_FDG_STAMP}" \
    --extra '{"bs":6}' --queue "seganypet.fdg_pretrain running,seganypet fs" \
    --updated-note "SegAnyPET FDG running"

  TASK1_SEGANY_FDG_BATCH_SIZE=6 \
  TASK1_NNUNET_RESULTS_STAMP_NAME="${SEG_FDG_STAMP}" \
    bash "${CTRL}/ICLR2026/run/run_seganypet_fdg_pretrain_bg.sh" \
    || _fail "SegAnyPET FDG failed"

  local SEG_BEST="${REPO}/runs/${SEG_FDG_STAMP}/seganypet_fdg/best.pth"
  [[ -f "${SEG_BEST}" ]] || SEG_BEST="${REPO}/runs/${SEG_FDG_STAMP}/seganypet_fdg/latest.pth"
  _require_file "${SEG_BEST}" "SegAnyPET FDG ckpt"

  local SEG_FS_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_fs50_from_fdg_f258_gpu013"
  _hb "seganypet_fs" "${SEG_FS_STAMP}"
  _board --set-stage seganypet.psma_fs50_f258 --status running --stamp "${SEG_FS_STAMP}" \
    --extra "{\"bs\":2,\"foundation\":\"${SEG_BEST}\"}" \
    --queue "seganypet.psma_fs50_f258 running" \
    --updated-note "SegAnyPET fewshot running"

  TASK1_SEGANY_CKPT="${SEG_BEST}" \
  TASK1_SEGANY_BATCH_SIZE=2 \
  TASK1_SEGANY_EPOCHS=100 \
  TASK1_SEGANY_ACCUM=20 \
  TASK1_SEGANY_LR_MODE=official \
  TASK1_SEGANY_CLICK_MAX=21 \
  TASK1_NNUNET_RESULTS_STAMP_NAME="${SEG_FS_STAMP}" \
    bash "${CTRL}/ICLR2026/run/run_seganypet_fewshot50_f258_bg.sh" \
    || _fail "SegAnyPET fewshot failed"

  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" --ingest-seganypet-stamp "${SEG_FS_STAMP}" || true
  _hb "seganypet_done" "${SEG_FS_STAMP}"
}

case "${FROM_STAGE}" in
  nnunet)
    run_nnunet
    run_monai
    run_seganypet
    ;;
  monai)
    run_monai
    run_seganypet
    ;;
  seganypet)
    run_seganypet
    ;;
  *)
    _fail "unknown FROM_STAGE=${FROM_STAGE}"
    ;;
esac

_hb "ALL_DONE" "${PIPE_STAMP}"
_board --updated-note "ALL STAGES DONE" --queue ""
echo "[aligned-stages] ALL DONE ${PIPE_STAMP}"
