#!/usr/bin/env bash
# Apply DpDNet PSMA dice-best checkpoints from existing logs + TEST20 refresh.
# SegAnyPET: already val_dice @ep100 for all folds — skip resume, optional TEST20 skip.
#
#   bash ICLR2026/run/run_dpdnet_apply_dice_best_and_test20_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"

DPD_PARENT="${DPD_PARENT:-20260817_210749_iclr2026_dpdnet_psma_fs50_f258_1gpu_bs2_tr25_val25e20_100ep_gpu013}"
SEG_STAMP="${SEG_STAMP:-20260817_114450_iclr2026_seganypet_fs50_from_fdg_f258_gpu013}"
FOLD_GPUS="${TASK1_FOLD_GPUS:-2:0,5:1,8:3}"
MANIFEST="${VIS}/dpdnet_dice_best_manifest_${DPD_PARENT}.json"
PIPE_LOG="${VIS}/nohup_dpdnet_apply_dice_best_and_test20.log"

exec > >(tee -a "${PIPE_LOG}") 2>&1
echo "[dice-best] === DpDNet PARENT=${DPD_PARENT} ==="

python3 "${CTRL}/ICLR2026/scripts/nnunet_pseudo_dice_best.py" \
  --parent-stamp "${DPD_PARENT}" \
  --folds 2,5,8 \
  --work "${WORK}" \
  --manifest "${MANIFEST}" \
  --apply || true

NEEDS="$(python3 "${CTRL}/ICLR2026/scripts/nnunet_pseudo_dice_best.py" \
  --parent-stamp "${DPD_PARENT}" --folds 2,5,8 --work "${WORK}" 2>/dev/null \
  | sed -n 's/^NEEDS_RETRAIN_FOLDS=//p' | tail -1)"

if [[ -n "${NEEDS}" ]]; then
  echo "[dice-best] retrain folds (${NEEDS}) → ep120 dice-best with ema_fg_dice"
  IFS=',' read -r -a _rf <<< "${NEEDS}"
  declare -A GPU_OF
  IFS=',' read -r -a _pairs <<< "${FOLD_GPUS}"
  for p in "${_pairs[@]}"; do GPU_OF["${p%%:*}"]="${p##*:}"; done
  for fold in "${_rf[@]}"; do
    gpu="${GPU_OF[${fold}]:-}"
    [[ -n "${gpu}" ]] || { echo "[error] no GPU for fold ${fold}" >&2; exit 1; }
    st="${DPD_PARENT}_f${fold}"
    fd="${WORK}/nnUNet_results/${st}/Dataset240_DpDNet_PSMA_2ch/STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres/fold_${fold}"
    final="${fd}/checkpoint_final.pth"
    [[ -f "${final}" ]] || { echo "[error] missing ${final}" >&2; exit 1; }
    docker run --rm --user root --entrypoint bash -v "${DATA}:${DATA}" autopet_baseline:latest -lc \
      "set -e; cp -a '${final}' '${fd}/checkpoint_latest.pth'; cp -a '${final}' '${fd}/checkpoint_best.pth'"
    rm -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${st}.txt"
    FOLD_ID="${fold}" GPU_ID="${gpu}" PARENT_STAMP="${DPD_PARENT}" \
      TASK1_DPDNET_NUM_EPOCHS=120 \
      TASK1_DPDNET_TRAIN_ITERS=25 \
      TASK1_DPDNET_VAL_ITERS=25 \
      TASK1_DPDNET_VAL_EVERY=20 \
      TASK1_BEST_BY=ema_fg_dice \
      TASK1_DPDNET_SKIP_PREPARE=1 \
      bash "${CTRL}/ICLR2026/run/run_dpdnet_psma_fewshot50_onefold_bg.sh"
    sleep 10
    echo "[dice-best] waiting fold${fold} STAMP=${st} until ep120 or stop"
    while true; do
      if [[ -f "${WORK}/01_train_vis/TASK1_TRAIN_STOP_${st}.txt" ]]; then
        echo "[dice-best] fold${fold} stopped"
        break
      fi
      if ! pgrep -af "${st}" >/dev/null 2>&1 \
         && ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qF "dpdnet_psma_f${fold}_${st}"; then
        echo "[dice-best] fold${fold} container exited"
        break
      fi
      sleep 45
    done
    TASK1_NNUNET_RESULTS_STAMP_NAME="${st}" bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
  done
fi

echo "[dice-best] === DpDNet TEST20 refresh (max val Dice ckpt) ==="
for fold in 2 5 8; do
  rm -f "${WORK}/nnUNet_results/${DPD_PARENT}/psma_test20_eval/fold${fold}/score_detail.json"
done
TASK1_FOLDS="2,5,8" TASK1_TEST_SKIP_DONE=0 TASK1_FOLD_GPUS="${FOLD_GPUS}" \
  PARENT_STAMP="${DPD_PARENT}" \
  bash "${CTRL}/ICLR2026/run/run_dpdnet_psma_test20_f258_parallel.sh"

echo "[dice-best] === SegAnyPET: skip resume (best val_dice @ep100 all folds) ==="
python3 - <<PY
import json
from pathlib import Path
stamp = "${SEG_STAMP}"
repo = Path("${CTRL}/ICLR2026/3D-MAE-PET-CT/runs")
rows = {}
for f in (2, 5, 8):
    m = repo / stamp / "seganypet" / f"fold{f}" / "metrics.jsonl"
    best_ep, best_v = None, -1.0
    for line in m.read_text().splitlines():
        r = json.loads(line)
        vd = r.get("val_dice")
        if vd is None or vd != vd:
            continue
        ep = int(r["epoch"])
        if float(vd) > best_v:
            best_v, best_ep = float(vd), ep
    rows[str(f)] = {"best_ep": best_ep, "best_val_dice": best_v, "resume_needed": False}
out = Path("${VIS}") / f"seganypet_dice_best_manifest_{stamp}.json"
out.write_text(json.dumps({"stamp": stamp, "policy": "best.pth = max val_dice", "folds": rows}, indent=2) + "\n")
print(out.read_text())
PY

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" \
  --patch-json "$(python3 - <<'PY'
import json
print(json.dumps({"updated_note": "DpDNet/SegAnyPET unified max val Dice best; DpDNet TEST20 refreshed"}))
PY
)" || true

echo "[dice-best] DONE log=${PIPE_LOG}"
