#!/usr/bin/env bash
# Remaining GPU schedule (GPUs 0,1,3):
#   1) 3-GPU jobs first (MIM fs10 TEST20 → fs5 TEST20)
#   2) 2-GPU + 1-GPU (encoder CT+PET retrain + fs50 fold2 TEST20)
#   3) leftover 1-GPU ×3 (MIM fc70 / fs0 / FDG TEST)
# CPU jobs are started separately and must not wait on this script.
#
#   bash ICLR2026/run/run_remaining_gpu_schedule_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
ENC_ROOT="${CTRL}/ICLR2026/3D-MAE-PET-CT/weights/dpdnet"
PID_FILE="${VIS}/remaining_gpu_schedule.pid"
LOG="${VIS}/nohup_remaining_gpu_schedule.log"
HOLD="${VIS}/TASK1_REMAINING_SCHEDULE_HOLD.txt"

MIM_FS50="20260829_155823_iclr2026_nnunet_psma_fs50_f258_1gpu_bs2_tr25_val25e20_100ep_gpu013"
MIM_FS10="20260830_201957_iclr2026_nnunet_mim_psma_fs10_f258_1gpu_bs2_tr25_val25e20_100ep_gpu013"
MIM_FS5="20260830_203725_iclr2026_nnunet_mim_psma_fs5_f258_1gpu_bs2_tr25_val25e20_100ep_gpu013"
MIM_FDG_STAMP="20260829_133121_iclr2026_nnunet_mim_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep"
MIM_FDG_CKPT="${WORK}/nnUNet_results/${MIM_FDG_STAMP}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

mkdir -p "${VIS}"
if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" && "${old}" != "$$" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[remain-sched] already running pid=${old}"
    exit 0
  fi
fi
echo $$ > "${PID_FILE}"
date -Iseconds > "${HOLD}"
exec > >(tee -a "${LOG}") 2>&1
echo "[remain-sched] $(date '+%F %T') pid=$$ HOLD=${HOLD}"

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD}"
export TASK1_BOARD_METHOD=nnunet_mim
export TASK1_UDA_FDG_STAMP="${MIM_FDG_STAMP}"
export TASK1_UDA_FDG_BEST="${MIM_FDG_CKPT}"
export TASK1_TEST_SKIP_DONE=1
export TASK1_UDA_PRED_PER_GPU="${TASK1_UDA_PRED_PER_GPU:-5}"

_board() {
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD}" --no-plot --patch-json "$1" || true
}

_disarm() {
  TASK1_NNUNET_RESULTS_STAMP_NAME="${1}" bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
}

_mim_test20() {
  local parent="$1" stage="$2" folds="${3:-2,5,8}" map="${4:-2:0,5:1,8:3}"
  _disarm "${parent}"
  _board "{\"methods\":{\"nnunet_mim\":{\"${stage}\":{\"status\":\"running\",\"stamp\":\"${parent}\",\"phase\":\"TEST20\",\"gpu_ids\":\"0,1,3\"}}},\"updated_note\":\"3GPU · MIM ${stage} TEST20\"}"
  PARENT_STAMP="${parent}" \
    TASK1_NNUNET_RESULTS_STAMP_NAME="${parent}" \
    TASK1_FOLDS="${folds}" \
    TASK1_FOLD_GPUS="${map}" \
    TASK1_BOARD_METHOD=nnunet_mim \
    TASK1_TEST_SKIP_DONE=1 \
    bash "${CTRL}/ICLR2026/run/run_nnunet_psma_test20_f258_parallel.sh"
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true
}

_encoders_ready() {
  [[ -f "${ENC_ROOT}/best_encoder_ct_epoch_94.pth" && -f "${ENC_ROOT}/best_encoder_pet_epoch_94.pth" ]]
}

_prep_ready() {
  [[ -f "${WORK}/nnUNet_preprocessed/Dataset250_DpDNet_FDG_CT1ch/nnUNetPlans.json" \
    && -f "${WORK}/nnUNet_preprocessed/Dataset251_DpDNet_FDG_PET1ch/nnUNetPlans.json" ]]
}

echo "[remain-sched] PHASE 3GPU-1 · MIM fs10 TEST20"
_mim_test20 "${MIM_FS10}" "psma_fs10_f258"

echo "[remain-sched] PHASE 3GPU-2 · MIM fs5 TEST20"
_mim_test20 "${MIM_FS5}" "psma_fs5_f258"

echo "[remain-sched] PHASE 2+1 · encoder CT+PET (2GPU) + fs50 fold2 TEST20 (1GPU)"
_disarm "${MIM_FS50}_f2"
_board "{\"methods\":{\"nnunet_mim\":{\"psma_fs50_f258\":{\"status\":\"running\",\"stamp\":\"${MIM_FS50}\",\"phase\":\"TEST20\",\"note\":\"fold2 remaining · GPU3\"}}},\"updated_note\":\"2+1 · encoders + fs50 fold2\"}"

PARENT_STAMP="${MIM_FS50}" \
  TASK1_NNUNET_RESULTS_STAMP_NAME="${MIM_FS50}" \
  TASK1_FOLDS=2 \
  TASK1_FOLD_GPUS=2:3 \
  TASK1_BOARD_METHOD=nnunet_mim \
  TASK1_TEST_SKIP_DONE=1 \
  bash "${CTRL}/ICLR2026/run/run_nnunet_psma_test20_f258_parallel.sh" &
p_fold2=$!

enc_pids=()
if _encoders_ready; then
  echo "[remain-sched] encoders already on disk — skip retrain"
else
  echo "[remain-sched] wait CPU encoder preprocess (fold2 already on GPU3)"
  waited=0
  while ! _prep_ready && [[ ! -f "${VIS}/TASK1_DPDNET_ENC_PREP_DONE.txt" ]]; do
    prep_pid="$(tr -d '[:space:]' < "${VIS}/dpdnet_dualenc_encoder_cpu_prep.pid" 2>/dev/null || true)"
    if [[ -n "${prep_pid}" ]] && ! kill -0 "${prep_pid}" 2>/dev/null; then
      echo "[remain-sched] encoder prep pid ${prep_pid} dead — skip wait, continue 1GPU"
      echo "status=dead" > "${VIS}/TASK1_DPDNET_ENC_PREP_DONE.txt"
      break
    fi
    if [[ "${waited}" -ge 1800 ]]; then
      echo "[remain-sched] encoder prep timeout 30m — continue 1GPU"
      echo "status=timeout" > "${VIS}/TASK1_DPDNET_ENC_PREP_DONE.txt"
      break
    fi
    sleep 30
    waited=$((waited + 30))
  done
  if _prep_ready && ! _encoders_ready; then
    MODALITY=ct GPU_ID=0 bash "${CTRL}/ICLR2026/run/run_dpdnet_dualenc_retrain_encoders.sh" &
    enc_pids+=($!)
    MODALITY=pet GPU_ID=1 bash "${CTRL}/ICLR2026/run/run_dpdnet_dualenc_retrain_encoders.sh" &
    enc_pids+=($!)
  fi
fi

wait "${p_fold2}" || echo "[remain-sched] fs50 fold2 TEST20 rc=$?"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true
for p in "${enc_pids[@]+"${enc_pids[@]}"}"; do
  wait "${p}" || echo "[remain-sched] encoder pid ${p} rc=$?"
done

echo "[remain-sched] PHASE 1GPU×3 · fc70 + fs0 + FDG TEST"
_disarm "nnunet_mim_1gpu_tail"
_board "{\"updated_note\":\"1GPU×3 · MIM fc70 / fs0 / FDG TEST\"}"

TASK1_BOARD_METHOD=nnunet_mim \
  TASK1_UDA_FDG_STAMP="${MIM_FDG_STAMP}" \
  TASK1_UDA_FDG_BEST="${MIM_FDG_CKPT}" \
  TASK1_PSMA_FC70_GPU=0 \
  TASK1_NNUNET_RESULTS_STAMP_NAME="" \
  bash "${CTRL}/ICLR2026/run/run_nnunet_psma_fc70_decline_and_test_bg.sh" &
p_fc70=$!

METHOD=nnunet_mim TASK1_TEST_SKIP_DONE=0 TASK1_CUDA_VISIBLE_DEVICES=1 \
  bash "${CTRL}/ICLR2026/run/run_eval_fdg_shared_test20_bg.sh" &
p_fs0=$!

METHOD=nnunet_mim TASK1_TEST_SKIP_DONE=0 TASK1_CUDA_VISIBLE_DEVICES=3 \
  bash "${CTRL}/ICLR2026/run/run_eval_fdg_test20_bg.sh" &
p_fdg=$!

wait "${p_fs0}" || echo "[remain-sched] fs0 rc=$?"
wait "${p_fdg}" || echo "[remain-sched] fdg_test rc=$?"
wait "${p_fc70}" || echo "[remain-sched] fc70 rc=$?"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true

if _encoders_ready; then
  echo "[remain-sched] dual-enc weights ready → FDG 1GPU then PSMA 3GPU"
  TASK1_DPDNET_GPU=0 \
    TASK1_BOARD_METHOD=dpdnet_dualenc \
    bash "${CTRL}/ICLR2026/run/run_dpdnet_dualenc_aligned_fdg_psma_bg.sh"
else
  echo "[remain-sched] dual-enc weights still missing — skip dualenc pipeline"
fi

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true
echo "[remain-sched] ALL DONE $(date '+%F %T')"
rm -f "${HOLD}"
