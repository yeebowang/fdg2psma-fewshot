#!/usr/bin/env bash
# DpDNet + PET/CT dual-encoder init (STUNetTrainer_small_prompt_pretrain)
#   FDG 169ep · 1GPU bs=6 · tr70/val0  (same protocol as scratch DpDNet)
#   PSMA fs50 f258 · 3GPU · tr25/val25 every20 · 100ep
#
# Requires:
#   ICLR2026/3D-MAE-PET-CT/weights/dpdnet/best_encoder_{ct,pet}_epoch_94.pth
#   or TASK1_DPDNET_CT_ENCODER / TASK1_DPDNET_PET_ENCODER
#
#   bash ICLR2026/run/run_dpdnet_dualenc_aligned_fdg_psma_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
ICLR_VIS="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"
ENC_ROOT="${CTRL}/ICLR2026/3D-MAE-PET-CT/weights/dpdnet"
CT_ENC="${TASK1_DPDNET_CT_ENCODER:-${ENC_ROOT}/best_encoder_ct_epoch_94.pth}"
PET_ENC="${TASK1_DPDNET_PET_ENCODER:-${ENC_ROOT}/best_encoder_pet_epoch_94.pth}"
IMAGE="${TASK1_NNUNET_IMAGE:-autopet_baseline:latest}"
mkdir -p "${ICLR_VIS}"
PID_FILE="${ICLR_VIS}/dpdnet_dualenc_aligned_fdg_psma.pid"
if [[ -f "${PID_FILE}" ]]; then
  old="$(tr -d '[:space:]' < "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old}" && "${old}" != "$$" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[dpdnet-dualenc] already running pid=${old}"
    exit 0
  fi
fi
echo $$ > "${PID_FILE}"

export TASK1_BASE="${DATA}"
export TASK1_ALIGN_BOARD_JSON="${BOARD_JSON}"
export TASK1_BOARD_METHOD=dpdnet_dualenc
export TRAINER=STUNetTrainer_small_prompt_pretrain
export TASK1_DPDNET_CT_ENCODER="${CT_ENC}"
export TASK1_DPDNET_PET_ENCODER="${PET_ENC}"

if [[ ! -f "${CT_ENC}" || ! -f "${PET_ENC}" ]]; then
  echo "[error] DpDNet PET+CT dual-encoder weights missing." >&2
  echo "  expected CT=${CT_ENC}" >&2
  echo "  expected PET=${PET_ENC}" >&2
  echo "  Previous crash cause: STUNetTrainer_small_prompt_pretrain hardcoded" >&2
  echo "  /projects/.../best_encoder_{ct,pet}_epoch_94.pth (not on this machine)" >&2
  echo "  and map_location=cuda during CPU-only network build." >&2
  exit 1
fi
_enc_width_ok() {
  docker run --rm --entrypoint python3 \
    -v "${CTRL}:${CTRL}" "${IMAGE}" \
    "${CTRL}/ICLR2026/scripts/check_dpdnet_encoder_width.py" \
    --ckpt "$1" --expect-out-ch 16
}
if ! _enc_width_ok "${CT_ENC}" || ! _enc_width_ok "${PET_ENC}"; then
  echo "[error] encoder ckpt is not 16-ch STUNet_prompt (dual-enc cannot load 32-ch STUNetTrainer_small)." >&2
  exit 1
fi

FDG_EP="${TASK1_DPDNET_NUM_EPOCHS:-169}"
FDG_TR="${TASK1_DPDNET_TRAIN_ITERS:-${TASK1_TRAIN_ITERS_PER_EPOCH:-70}}"
FDG_VAL="${TASK1_DPDNET_VAL_ITERS:-${TASK1_VAL_ITERS_PER_EPOCH:-0}}"
PSMA_EP="${TASK1_DPDNET_PSMA_EPOCHS:-100}"
PSMA_TR="${TASK1_DPDNET_PSMA_TRAIN_ITERS:-25}"
PSMA_VAL="${TASK1_DPDNET_PSMA_VAL_ITERS:-25}"
PSMA_EVERY="${TASK1_DPDNET_VAL_EVERY:-20}"
PSMA_BS="${TASK1_DPDNET_BATCH_SIZE:-2}"
GPU_ID="${TASK1_DPDNET_GPU:-${TASK1_CUDA_VISIBLE_DEVICES:-0}}"
GPU_ID="${GPU_ID%%,*}"

PIPE_LOG="${ICLR_VIS}/nohup_dpdnet_dualenc_aligned_fdg${FDG_EP}_psma_f258.log"
exec > >(tee -a "${PIPE_LOG}") 2>&1
echo "[dpdnet-dualenc] FDG ${FDG_EP}ep dual-enc init → PSMA fs50"

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"dpdnet_dualenc\":{\"fdg_pretrain\":{\"status\":\"running\",\"total_epochs\":${FDG_EP},\"train_iters\":${FDG_TR},\"val_iters\":${FDG_VAL},\"bs\":6,\"note\":\"PET+CT dual-enc → FDG tr${FDG_TR}/val${FDG_VAL} · ${FDG_EP}ep\"}}},\"updated_note\":\"DpDNet dual-enc FDG running\"}" || true

export TASK1_DPDNET_NUM_EPOCHS="${FDG_EP}"
export TASK1_DPDNET_TRAIN_ITERS="${FDG_TR}"
export TASK1_DPDNET_VAL_ITERS="${FDG_VAL}"
export TASK1_DPDNET_BATCH_SIZE=6
export TASK1_DPDNET_GPU="${GPU_ID}"
export TASK1_DPDNET_SKIP_PREPARE="${TASK1_DPDNET_SKIP_PREPARE:-0}"
export TASK1_DPDNET_SKIP_ENCODER_INIT=0
export TASK1_DPDNET_LAST_STAMP_FILE="${ICLR_VIS}/dpdnet_dualenc_fdg_LAST_STAMP.txt"
export TASK1_BEST_BY=train_loss
unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
bash "${CTRL}/ICLR2026/run/run_dpdnet_fdg_1gpu_bs6_bg.sh"

FDG_STAMP="$(tr -d '[:space:]' < "${ICLR_VIS}/dpdnet_dualenc_fdg_LAST_STAMP.txt")"
[[ -n "${FDG_STAMP}" ]] || { echo "[error] no DpDNet dualenc FDG stamp" >&2; exit 1; }

FDG_DS="Dataset239_DpDNet_FDG_2ch"
FDG_TF="STUNetTrainer_small_prompt_pretrain__nnUNetPlans__3d_fullres"
FDG_FOLD="${WORK}/nnUNet_results/${FDG_STAMP}/${FDG_DS}/${FDG_TF}/fold_0"
FDG_FINAL="${FDG_FOLD}/checkpoint_final.pth"
CNAME="dpdnet_fdg_${FDG_STAMP}"

_resolve_dpd_fdg_ckpt() {
  local f
  for f in checkpoint_final.pth checkpoint_latest.pth checkpoint_best.pth; do
    if [[ -f "${FDG_FOLD}/${f}" ]]; then
      echo "${FDG_FOLD}/${f}"
      return 0
    fi
  done
  return 1
}

echo "[dpdnet-dualenc] wait FDG ${FDG_STAMP}…"
_fdg_miss=0
while [[ ! -f "${FDG_FINAL}" ]]; do
  if docker ps --format '{{.Names}}' | grep -qx "${CNAME}"; then
    echo "[dpdnet-dualenc] FDG still running $(TZ=Asia/Shanghai date +%H:%M:%S)"
    _fdg_miss=0
    sleep 90
    continue
  fi
  if _resolve_dpd_fdg_ckpt >/dev/null; then
    break
  fi
  _fdg_miss=$((_fdg_miss + 1))
  if [[ "${_fdg_miss}" -ge 4 ]]; then
    echo "[error] dualenc FDG container gone and no ckpt (${CNAME})" >&2
    exit 1
  fi
  echo "[dpdnet-dualenc] FDG not up yet (${_fdg_miss}/4) $(TZ=Asia/Shanghai date +%H:%M:%S)"
  sleep 30
done
FDG_CKPT="$(_resolve_dpd_fdg_ckpt)" || true
[[ -n "${FDG_CKPT}" && -f "${FDG_CKPT}" ]] || { echo "[error] missing dualenc FDG ckpt" >&2; exit 1; }
echo "[dpdnet-dualenc] FDG done; PSMA init=${FDG_CKPT}"

TASK1_NNUNET_RESULTS_STAMP_NAME="${FDG_STAMP}" bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"dpdnet_dualenc\":{\"fdg_pretrain\":{\"status\":\"done\",\"stamp\":\"${FDG_STAMP}\",\"best_ckpt\":\"${FDG_CKPT}\",\"total_epochs\":${FDG_EP},\"note\":\"PET+CT dual-enc → FDG done · ${FDG_EP}ep\"},\"psma_fs50_f258\":{\"status\":\"running\"}}},\"updated_note\":\"DpDNet dual-enc FDG done → PSMA\"}" || true

export TASK1_DPDNET_FDG_STAMP="${FDG_STAMP}"
export TASK1_DPDNET_FDG_BEST="${FDG_CKPT}"
export TASK1_DPDNET_FDG_FORCE_STAMP=1
export TASK1_DPDNET_FDG_TF="${FDG_TF}"
export TASK1_DPDNET_NUM_EPOCHS="${PSMA_EP}"
export TASK1_DPDNET_TRAIN_ITERS="${PSMA_TR}"
export TASK1_DPDNET_VAL_ITERS="${PSMA_VAL}"
export TASK1_DPDNET_VAL_EVERY="${PSMA_EVERY}"
export TASK1_DPDNET_BATCH_SIZE="${PSMA_BS}"
export TASK1_BEST_BY=val_loss
export TASK1_FOLDS=2,5,8
export TASK1_FOLD_GPUS=2:0,5:1,8:3
export TASK1_TEST_SKIP_DONE=1
export TASK1_BOARD_METHOD=dpdnet_dualenc
export TRAINER=STUNetTrainer_small_prompt_pretrain
# PSMA loads FDG ckpt via -pretrained_weights; skip re-reading epoch_94 encoders.
export TASK1_DPDNET_SKIP_ENCODER_INIT=1
unset TASK1_NNUNET_RESULTS_STAMP_NAME || true

bash "${CTRL}/ICLR2026/run/run_dpdnet_psma_fewshot50_f258_1gpu_bs2_100ep_bg.sh"
python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" --board "${BOARD_JSON}" || true
# fs50 后自动接 fs10→fs5→fc70/fs0/FDG TEST（避免 GPU 空转）
if [[ "${TASK1_DPDNET_DUALENC_CHAIN_CONTINUE:-1}" == "1" ]]; then
  echo "[dpdnet-dualenc] chaining continue_after_fs50…"
  bash "${CTRL}/ICLR2026/run/run_dpdnet_dualenc_continue_after_fs50_bg.sh"
else
  echo "[dpdnet-dualenc] fs50 DONE (chain continue skipped)"
fi
