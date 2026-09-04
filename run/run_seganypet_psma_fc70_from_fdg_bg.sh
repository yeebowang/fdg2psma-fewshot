#!/usr/bin/env bash
# SegAnyPET PSMA fc70% · single GPU · fold0 · 100ep → decline → TEST20
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/ICLR2026/run/_psma_fc70_env.sh"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"
SEG_CODE="${CTRL}/ICLR2026/third_party/SegAnyPET/code"
SEG_PIP="${CTRL}/ICLR2026/third_party/seganypet_pip"
MON="${CTRL}/ICLR2026/scripts/monitor_val_dice_decline_stop.py"
RAW_ROOT="${TASK1_SEGANY_RAW_ROOT:-${DATA}/task1_train_workspace/nnUNet_raw/Dataset221_AutoPETIV_Task1_4ch}"
DATA_ROOT="${TASK1_SEGANY_DATA_ROOT:-${DATA}/task1_train_workspace/seganypet_psma_fc70}"
WEIGHT_DIR="${REPO}/weights/seganypet"
BOARD_METHOD="${TASK1_BOARD_METHOD:-seganypet}"

read -r TR VAL NTR NVAL < <(_fc70_resolve_iters)

SEG_STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME:-}"
[[ -n "${SEG_STAMP}" && "${SEG_STAMP}" == *fc70* ]] || {
  if [[ "${BOARD_METHOD}" == "seganypet_scratch" ]]; then
    SEG_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_scratch_psma_fc70_1gpu_bs${PSMA_FC70_BS}_tr${TR}_val${VAL}e${PSMA_FC70_VAL_EVERY}_${PSMA_FC70_EP}ep_gpu${PSMA_FC70_GPU}"
  else
    SEG_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_seganypet_psma_fc70_1gpu_bs${PSMA_FC70_BS}_tr${TR}_val${VAL}e${PSMA_FC70_VAL_EVERY}_${PSMA_FC70_EP}ep_gpu${PSMA_FC70_GPU}"
  fi
}

PIPE_LOG="${VIS}/nohup_seganypet_psma_fc70_${SEG_STAMP}.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1
echo "[seganypet-fc70] STAMP=${SEG_STAMP} n=${NTR}/${NVAL} gpu=${PSMA_FC70_GPU} fold=${PSMA_FC70_FOLD}"

if [[ -n "${TASK1_SEGANY_CKPT:-}" ]]; then
  CKPT="${TASK1_SEGANY_CKPT}"
elif [[ "${BOARD_METHOD}" == "seganypet_scratch" ]]; then
  echo "[error] seganypet_scratch fc70 needs TASK1_SEGANY_CKPT (FDG scratch best.pth)" >&2
  exit 1
elif [[ -f "${WEIGHT_DIR}/seganypet_lesion.pth" ]]; then
  CKPT="${WEIGHT_DIR}/seganypet_lesion.pth"
else
  CKPT="${WEIGHT_DIR}/seganypet_v2.pth"
fi
[[ -f "${CKPT}" ]] || { echo "[error] missing SegAnyPET ckpt ${CKPT}" >&2; exit 1; }

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"${PSMA_FC70_STAGE}\":{\"status\":\"running\",\"stamp\":\"${SEG_STAMP}\",\"note\":\"fc70% single run\"}}},\"updated_note\":\"${BOARD_METHOD} fc70 training\"}" || true

bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true

python3 - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${CTRL}/ICLR2026/scripts")
from prepare_seganypet_fewshot_f258 import export_fold
export_fold(
    ${PSMA_FC70_FOLD},
    Path("${PSMA_FC70_SPLITS}"),
    Path("${RAW_ROOT}"),
    Path("${DATA_ROOT}"),
)
PY

OUT_ROOT="${REPO}/runs/${SEG_STAMP}"
fold_data="${DATA_ROOT}/fold${PSMA_FC70_FOLD}"
out_dir="${OUT_ROOT}/seganypet/fold${PSMA_FC70_FOLD}"
mkdir -p "${out_dir}"
cname="seganypet_fc70_f${PSMA_FC70_FOLD}_${SEG_STAMP}"
log="${VIS}/nohup_seganypet_fc70_train_${SEG_STAMP}.log"

docker rm -f "${cname}" >/dev/null 2>&1 || true
STOP="${WORK}/01_train_vis/TASK1_TRAIN_STOP_${SEG_STAMP}.txt"
rm -f "${STOP}" || true

nohup python3 "${MON}" --method seganypet --parent-stamp "${SEG_STAMP}" --fold "${PSMA_FC70_FOLD}" \
  --base-ep "${PSMA_FC70_BASE_EP}" --val-every "${PSMA_FC70_VAL_EVERY}" \
  >"${VIS}/nohup_decline_mon_seganypet_fc70.log" 2>&1 &

echo "[seganypet-fc70] train 100ep foreground → ${log}"
docker run --rm \
  --name "${cname}" \
  --gpus "device=${PSMA_FC70_GPU}" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONPATH="${SEG_PIP}:${SEG_CODE}:${CTRL}/ICLR2026/scripts" \
  -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
  -w "${SEG_CODE}" --shm-size=8g "${IMAGE}" \
  python3 "${CTRL}/ICLR2026/scripts/seganypet_fewshot_finetune.py" \
    --data-root "${fold_data}" \
    --checkpoint "${CKPT}" \
    --out-dir "${out_dir}" \
    --epochs "${PSMA_FC70_EP}" \
    --batch-size "${PSMA_FC70_BS}" --accumulation-steps 4 --num-workers 6 \
    --val-interval "${PSMA_FC70_VAL_EVERY}" --val-clicks 5 --val-max-cases 15 \
    --lr-mode finetune --lr 8e-4 --milestones 60,85 --click-max 11 --no-dataparallel --fresh \
  >>"${log}" 2>&1 || true

if [[ ! -f "${STOP}" ]]; then
  echo "[seganypet-fc70] continue until decline / max ${PSMA_FC70_MAX_EP}ep"
  docker rm -f "${cname}" >/dev/null 2>&1 || true
  docker run --rm \
    --name "${cname}" \
    --gpus "device=${PSMA_FC70_GPU}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONPATH="${SEG_PIP}:${SEG_CODE}:${CTRL}/ICLR2026/scripts" \
    -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
    -w "${SEG_CODE}" --shm-size=8g "${IMAGE}" \
    python3 "${CTRL}/ICLR2026/scripts/seganypet_fewshot_finetune.py" \
      --data-root "${fold_data}" \
      --checkpoint "${CKPT}" \
      --out-dir "${out_dir}" \
      --epochs "${PSMA_FC70_MAX_EP}" \
      --batch-size "${PSMA_FC70_BS}" --accumulation-steps 4 --num-workers 6 \
      --val-interval "${PSMA_FC70_VAL_EVERY}" --val-clicks 5 --val-max-cases 15 \
      --lr-mode finetune --lr 8e-4 --milestones 60,85 --click-max 11 --no-dataparallel \
    >>"${log}" 2>&1 || true
fi

if [[ ! -f "${out_dir}/best.pth" && ! -f "${out_dir}/latest.pth" ]]; then
  echo "[error] no seganypet ckpt after train (${out_dir}) — skip TEST20" >&2
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD}" --no-plot \
    --patch-json "{\"methods\":{\"seganypet\":{\"${PSMA_FC70_STAGE}\":{\"status\":\"pending\",\"note\":\"defer · extra-fold 9fold first (fc70 TEST20 missing ckpt)\"}}},\"updated_note\":\"SegAnyPET fc70 skipped TEST20 (no ckpt)\"}" || true
  exit 1
fi

TASK1_PSMA_BOARD_STAGE="${PSMA_FC70_STAGE}" TASK1_FEWSHOT_N=70 \
  TASK1_SEGANY_FOLDS_CSV="${PSMA_FC70_FOLD}" TASK1_FOLD_GPUS="${PSMA_FC70_FOLD}:${PSMA_FC70_GPU}" \
  TASK1_TEST_SKIP_DONE=0 \
  STAMP="${SEG_STAMP}" bash "${CTRL}/ICLR2026/run/run_eval_seganypet_psma_test20_f258_bg.sh"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json "{\"methods\":{\"seganypet\":{\"${PSMA_FC70_STAGE}\":{\"status\":\"done\",\"stamp\":\"${SEG_STAMP}\",\"note\":\"TEST20 DONE · fc70 single\"}}},\"updated_note\":\"SegAnyPET fc70 TEST20 done\"}" || true

echo "[seganypet-fc70] ALL DONE ${SEG_STAMP}"
