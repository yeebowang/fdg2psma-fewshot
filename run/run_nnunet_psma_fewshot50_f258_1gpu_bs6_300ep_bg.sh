#!/usr/bin/env bash
# nnUNet PSMA fewshot50 finetune · folds 2/5/8 · 1 GPU/fold · tr=70 · val=70 every 20ep · 300ep
# Init: FDG checkpoint_final>latest（aligned val0）；PSMA best 按 val_loss；训完后 TEST20。
#
#   export TASK1_BASE=/media/ybwang/data1/PSMA-DATA
#   bash ICLR2026/run/run_nnunet_psma_fewshot50_f258_1gpu_bs6_300ep_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK1_REPO_ROOT="${ROOT}"
export TASK1_BASE="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
export WORK_DIR="${WORK_DIR:-${TASK1_BASE}/task1_train_workspace}"
WORK="${WORK_DIR}"
ICLR_VIS="${TASK1_LOSS_OUT_DIR:-${ROOT}/ICLR2026/vis}"
mkdir -p "${ICLR_VIS}"

DATASET_ID="${DATASET_ID:-228}"
DS="Dataset${DATASET_ID}_AutoPETIV_Task1_2ch"
TRAINER="${TRAINER:-nnUNetTrainer_Task1StdTrainVal50}"
PLANS_ID="${PLANS_ID:-nnUNetPlans}"
CONFIG="${CONFIG:-3d_fullres}"
TF="${TRAINER}__${PLANS_ID}__${CONFIG}"
TOTAL_EPOCHS="${TASK1_NUM_EPOCHS:-300}"
TRAIN_ITERS="${TASK1_TRAIN_ITERS_PER_EPOCH:-70}"
# 默认对齐 MAE：val_iters=70，每 20 ep 跑一次（末轮必跑）
VAL_ITERS="${TASK1_FS50_VAL_ITERS:-70}"
VAL_EVERY="${TASK1_FS50_VAL_EVERY_N_EPOCHS:-20}"
BATCH="${TASK1_FIXED_BATCH_3D_FULLRES:-6}"
FOLDS_CSV="${TASK1_FOLDS:-2,5,8}"
FOLD_GPUS_CSV="${TASK1_FOLD_GPUS:-2:0,5:1,8:3}"
FEWSHOT_N="${TASK1_FEWSHOT_N:-50}"
BOARD_STAGE="${TASK1_PSMA_BOARD_STAGE:-psma_fs${FEWSHOT_N}_f258}"
BOARD_METHOD="${TASK1_BOARD_METHOD:-nnunet}"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
export TASK1_FEWSHOT_SPLIT_DIR="${TASK1_FEWSHOT_SPLIT_DIR:-${ROOT}/ICLR2026/data/splits_mae_psma_fewshot${FEWSHOT_N}_9fold}"

FDG_STAMP="${TASK1_UDA_FDG_STAMP:-20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep}"
FDG_BEST="${TASK1_UDA_FDG_BEST:-}"
if [[ -z "${FDG_BEST}" ]]; then
  _fold="${WORK}/nnUNet_results/${FDG_STAMP}/${DS}/${TF}/fold_0"
  for _c in checkpoint_final.pth checkpoint_latest.pth checkpoint_best.pth; do
    [[ -f "${_fold}/${_c}" ]] && { FDG_BEST="${_fold}/${_c}"; break; }
  done
fi
[[ -n "${FDG_BEST}" && -f "${FDG_BEST}" ]] || { echo "[error] missing FDG final/latest: ${FDG_BEST:-<unset>}" >&2; exit 1; }
PREP_PLANS="${WORK}/nnUNet_preprocessed/${DS}/nnUNetPlans.json"
[[ -f "${PREP_PLANS}" ]] || { echo "[error] missing ${PREP_PLANS}" >&2; exit 1; }

_req="${TASK1_NNUNET_RESULTS_STAMP_NAME:-}"
_psma_tag="nnunet"
if [[ "${BOARD_METHOD}" =~ ^(hemingduo_scratch|chenyixin_scratch|hemingduo|chenyixin)$ ]]; then
  _psma_tag="${BOARD_METHOD}"
fi
if [[ -n "${_req}" && "${_req}" == *${_psma_tag}_psma_fs${FEWSHOT_N}* && "${_req}" != *_f[0-9] ]]; then
  PARENT="${_req}"
else
  PARENT="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_${_psma_tag}_psma_fs${FEWSHOT_N}_f258_1gpu_bs${BATCH}_tr${TRAIN_ITERS}_val${VAL_ITERS}e${VAL_EVERY}_${TOTAL_EPOCHS}ep_gpu013"
fi

IFS=',' read -r -a FOLDS <<< "${FOLDS_CSV}"
declare -A GPU_OF
IFS=',' read -r -a _pairs <<< "${FOLD_GPUS_CSV}"
for p in "${_pairs[@]}"; do
  GPU_OF["${p%%:*}"]="${p##*:}"
done

AGG_ROOT="${WORK}/nnUNet_results/${PARENT}"
PIPE_LOG="${ICLR_VIS}/nohup_nnunet_psma_fs${FEWSHOT_N}_f258_${PARENT}.log"
META="${ICLR_VIS}/iclr2026_nnunet_psma_fs${FEWSHOT_N}_f258_${PARENT}.txt"
mkdir -p "${AGG_ROOT}"
exec > >(tee -a "${PIPE_LOG}") 2>&1

{
  echo "job=iclr2026_nnunet_psma_fewshot${FEWSHOT_N}_f258_1gpu"
  echo "PARENT=${PARENT}"
  echo "folds=${FOLDS_CSV} gpus=${FOLD_GPUS_CSV}"
  echo "epochs=${TOTAL_EPOCHS} tr=${TRAIN_ITERS} val=${VAL_ITERS} every=${VAL_EVERY} bs=${BATCH}"
  echo "best_by=${TASK1_BEST_BY:-val_loss} val_loss_only=${TASK1_VAL_LOSS_ONLY:-1} pretrained=${FDG_BEST}"
} | tee "${META}"

echo "[nnunet-fs${FEWSHOT_N}] PARENT=${PARENT} stage=${BOARD_STAGE}"

python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" --no-plot \
  --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"${BOARD_STAGE}\":{\"status\":\"running\",\"stamp\":\"${PARENT}\",\"epoch\":0,\"total_epochs\":${TOTAL_EPOCHS},\"train_iters\":${TRAIN_ITERS},\"val_iters\":${VAL_ITERS},\"online_val\":\"VAL${VAL_ITERS} every${VAL_EVERY}\",\"bs\":${BATCH},\"test_invalidated\":true,\"fold_dice\":{},\"mean\":null,\"fold_ckpt_ep\":{},\"note\":\"tr${TRAIN_ITERS}/val${VAL_ITERS}e${VAL_EVERY} · ${TOTAL_EPOCHS}ep · fs${FEWSHOT_N}\"}}},\"updated_note\":\"${BOARD_METHOD} PSMA fs${FEWSHOT_N} training\"}" || true

# Set plans batch once
python3 - <<PY
import json
from pathlib import Path
p = Path("${PREP_PLANS}")
d = json.loads(p.read_text())
n = int("${BATCH}")
cfg = d["configurations"]["3d_fullres"]
old = cfg.get("batch_size")
cfg["batch_size"] = n
p.write_text(json.dumps(d, indent=2) + "\n")
print(f"[nnunet-fs50] plans batch_size {old} -> {n}")
PY

for fold in "${FOLDS[@]}"; do
  gpu="${GPU_OF[${fold}]:-}"
  [[ -n "${gpu}" ]] || { echo "[error] no GPU for fold ${fold}" >&2; exit 1; }
  echo "[nnunet-fs50] launch fold${fold} → GPU${gpu}"
  env -u TASK1_NNUNET_RESULTS_STAMP_NAME \
    FOLD_ID="${fold}" GPU_ID="${gpu}" PARENT_STAMP="${PARENT}" \
    TASK1_BASE="${TASK1_BASE}" \
    TASK1_NUM_EPOCHS="${TOTAL_EPOCHS}" \
    TASK1_TRAIN_ITERS_PER_EPOCH="${TRAIN_ITERS}" \
    TASK1_VAL_ITERS_PER_EPOCH="${VAL_ITERS}" \
    TASK1_VAL_EVERY_N_EPOCHS="${VAL_EVERY}" \
    TASK1_FS50_VAL_ITERS="${VAL_ITERS}" \
    TASK1_FS50_VAL_EVERY_N_EPOCHS="${VAL_EVERY}" \
    TASK1_BEST_BY="${TASK1_BEST_BY:-val_loss}" \
    TASK1_VAL_LOSS_ONLY="${TASK1_VAL_LOSS_ONLY:-1}" \
    TASK1_FIXED_BATCH_3D_FULLRES="${BATCH}" \
    TASK1_FEWSHOT_N="${FEWSHOT_N}" \
    TASK1_FEWSHOT_SPLIT_DIR="${TASK1_FEWSHOT_SPLIT_DIR}" \
    TASK1_UDA_FDG_BEST="${FDG_BEST}" \
    TASK1_UDA_FDG_STAMP="${FDG_STAMP}" \
    TASK1_DOCKER_BACKGROUND=1 \
    bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fewshot50_onefold_bg.sh" &
  echo $! >"${AGG_ROOT}/launch_fold${fold}.pid"
  sleep 12
done

echo "[nnunet-fs${FEWSHOT_N}] waiting for fold trains to finish (ep=${TOTAL_EPOCHS})…"

_fold_epoch() {
  local fold="$1"
  local stamp="${PARENT}_f${fold}"
  local fold_dir="${WORK}/nnUNet_results/${stamp}/${DS}/${TF}/fold_0"
  python3 - <<PY
from pathlib import Path
import re
fd = Path("${fold_dir}")
if (fd / "checkpoint_final.pth").is_file():
    print(${TOTAL_EPOCHS})
    raise SystemExit
logs = sorted(fd.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
ep = 0
if logs:
    for line in logs[0].read_text(errors="ignore").splitlines():
        for pat in (r"Epoch[: ]+(\d+)", r"current epoch[: ]+(\d+)", r"epoch[: ]+(\d+)"):
            m = re.search(pat, line, re.I)
            if m:
                ep = max(ep, int(m.group(1)))
print(ep)
PY
}

while true; do
  ok=0
  for fold in "${FOLDS[@]}"; do
    ep="$(_fold_epoch "${fold}")"
    echo "[nnunet-fs${FEWSHOT_N}] fold${fold} ep=${ep}/${TOTAL_EPOCHS}"
    [[ "${ep}" -ge "${TOTAL_EPOCHS}" ]] && ok=$((ok + 1))
  done
  [[ "${ok}" -ge "${#FOLDS[@]}" ]] && break
  sleep 90
done

if [[ "${TASK1_SKIP_TEST20_AT_END:-0}" != "1" ]]; then
echo "[nnunet-fs${FEWSHOT_N}] all folds trained → PSMA TEST20 Dice (1 GPU/fold · ${TASK1_UDA_PRED_PER_GPU:-5} shards/GPU)"

for fold in "${FOLDS[@]}"; do
  TASK1_CRASH_MONITOR_STAGE="nnunet_fs${FEWSHOT_N}_f${fold}_before_eval" \
  TASK1_NNUNET_RESULTS_STAMP_NAME="${PARENT}_f${fold}" \
    bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
done

export PARENT_STAMP="${PARENT}"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${PARENT}"
export TASK1_FOLDS="${FOLDS_CSV}"
export TASK1_FOLD_GPUS="${FOLD_GPUS_CSV}"
export TASK1_UDA_PRED_PER_GPU="${TASK1_UDA_PRED_PER_GPU:-5}"
export TASK1_TEST_SKIP_DONE=1
bash "${ROOT}/ICLR2026/run/run_nnunet_psma_test20_f258_parallel.sh"

AGG_JSON="${AGG_ROOT}/aggregate_test20_dice_f258.json"
echo "aggregate=${AGG_JSON}" >>"${META}"
echo "[nnunet-fs${FEWSHOT_N}] ALL DONE TEST20 → ${AGG_JSON}"
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --no-plot --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"${BOARD_STAGE}\":{\"status\":\"done\",\"test_invalidated\":false,\"phase\":null,\"note\":\"TEST20 DONE · 3/3\"}}},\"updated_note\":\"${BOARD_METHOD} PSMA fs${FEWSHOT_N} TEST20 done\"}" || true
else
  echo "[nnunet-fs${FEWSHOT_N}] skip TEST20 at end (TASK1_SKIP_TEST20_AT_END=1) PARENT=${PARENT}"
  echo "skip_test20=1" >>"${META}"
fi
