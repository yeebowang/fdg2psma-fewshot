#!/usr/bin/env bash
# Resume remaining schedule from 1GPU×3 (fs10/fs5/fs50 TEST20 already done).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIS="${ROOT}/ICLR2026/vis"
BOARD="${TASK1_ALIGN_BOARD_JSON:-${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
PID_FILE="${VIS}/remaining_1gpu_tail.pid"
LOG="${VIS}/nohup_remaining_1gpu_tail.log"
ENC_ROOT="${ROOT}/ICLR2026/3D-MAE-PET-CT/weights/dpdnet"
MIM_FDG_STAMP="20260829_133121_iclr2026_nnunet_mim_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep"
MIM_FDG_CKPT="/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/${MIM_FDG_STAMP}/Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" && "${old}" != "$$" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[1gpu-tail] already running pid=${old}"
    exit 0
  fi
fi
echo $$ > "${PID_FILE}"
exec > >(tee -a "${LOG}") 2>&1
echo "[1gpu-tail] $(date '+%F %T') start pid=$$"

bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD}" --no-plot \
  --patch-json '{"updated_note":"1GPU×3 · MIM fc70 / fs0 / FDG TEST"}' || true

TASK1_BOARD_METHOD=nnunet_mim \
  TASK1_UDA_FDG_STAMP="${MIM_FDG_STAMP}" \
  TASK1_UDA_FDG_BEST="${MIM_FDG_CKPT}" \
  TASK1_PSMA_FC70_GPU=0 \
  bash "${ROOT}/ICLR2026/run/run_nnunet_psma_fc70_decline_and_test_bg.sh" &
p_fc70=$!

METHOD=nnunet_mim TASK1_TEST_SKIP_DONE=0 TASK1_CUDA_VISIBLE_DEVICES=1 \
  bash "${ROOT}/ICLR2026/run/run_eval_fdg_shared_test20_bg.sh" &
p_fs0=$!

METHOD=nnunet_mim TASK1_TEST_SKIP_DONE=0 TASK1_CUDA_VISIBLE_DEVICES=3 \
  bash "${ROOT}/ICLR2026/run/run_eval_fdg_test20_bg.sh" &
p_fdg=$!

wait "${p_fs0}" || echo "[1gpu-tail] fs0 rc=$?"
wait "${p_fdg}" || echo "[1gpu-tail] fdg rc=$?"
wait "${p_fc70}" || echo "[1gpu-tail] fc70 rc=$?"
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true

if [[ -f "${ENC_ROOT}/best_encoder_ct_epoch_94.pth" && -f "${ENC_ROOT}/best_encoder_pet_epoch_94.pth" ]]; then
  echo "[1gpu-tail] encoders ready → dualenc FDG→PSMA"
  TASK1_DPDNET_GPU=0 TASK1_BOARD_METHOD=dpdnet_dualenc \
    bash "${ROOT}/ICLR2026/run/run_dpdnet_dualenc_aligned_fdg_psma_bg.sh"
else
  echo "[1gpu-tail] encoders not ready — dualenc later"
fi
python3 "${ROOT}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD}" || true
echo "[1gpu-tail] DONE $(date '+%F %T')"
rm -f /media/ybwang/data1/PSMA-CTRL/ICLR2026/vis/TASK1_REMAINING_SCHEDULE_HOLD.txt
