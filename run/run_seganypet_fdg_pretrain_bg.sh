#!/usr/bin/env bash
# SegAnyPET · FDG supervised click pretrain (align nnUNet Baseline1 FDG stage).
# Init: seganypet_lesion.pth → train on FDG 70% (PET-only) → best.pth for PSMA fewshot.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"
SEG_CODE="${CTRL}/ICLR2026/third_party/SegAnyPET/code"
SEG_PIP="${CTRL}/ICLR2026/third_party/seganypet_pip"
WEIGHT_DIR="${REPO}/weights/seganypet"

EPOCHS="${TASK1_SEGANY_FDG_EPOCHS:-100}"
BATCH_SIZE="${TASK1_SEGANY_FDG_BATCH_SIZE:-6}"
ACCUM="${TASK1_SEGANY_FDG_ACCUM:-20}"
GPU_DEVICES="${TASK1_SEGANY_OFFICIAL_GPUS:-0,1,3}"
# Inside container: map N docker GPUs → 0..N-1
if [[ -n "${TASK1_INNER_CUDA_VISIBLE_DEVICES:-}" ]]; then
  INNER_CVD="${TASK1_INNER_CUDA_VISIBLE_DEVICES}"
else
  _ng="$(awk -F',' '{print NF}' <<<"${GPU_DEVICES}")"
  INNER_CVD="$(seq -s, 0 $((_ng - 1)))"
fi
if [[ -n "${TASK1_PREFLIGHT_GPUS:-}" ]]; then
  PREFLIGHT_GPUS="${TASK1_PREFLIGHT_GPUS}"
else
  PREFLIGHT_GPUS="${GPU_DEVICES//,/ }"
fi
_ngpu="$(awk -F',' '{print NF}' <<<"${GPU_DEVICES}")"
WORKERS_PER_GPU="${TASK1_SEGANY_WORKERS_PER_GPU:-6}"
if [[ -n "${TASK1_SEGANY_WORKERS:-}" ]]; then
  WORKERS="${TASK1_SEGANY_WORKERS}"
else
  WORKERS="$((WORKERS_PER_GPU * _ngpu))"
fi
VAL_INTERVAL="${TASK1_SEGANY_VAL_INTERVAL:-20}"
VAL_CLICKS="${TASK1_SEGANY_VAL_CLICKS:-5}"
VAL_MAX_CASES="${TASK1_SEGANY_VAL_MAX_CASES:-15}"
LR="${TASK1_SEGANY_OFFICIAL_LR:-8e-4}"
# OOM fallback 6→3 unless TASK1_SEGANY_FDG_NO_OOM_FALLBACK=1
ALLOW_OOM_FALLBACK="${TASK1_SEGANY_FDG_NO_OOM_FALLBACK:-0}"

BOARD_METHOD="${TASK1_BOARD_METHOD:-seganypet}"
if [[ -n "${TASK1_SEGANY_CKPT:-}" ]]; then
  CKPT="${TASK1_SEGANY_CKPT}"
elif [[ "${BOARD_METHOD}" == "seganypet_scratch" ]]; then
  CKPT="none"
elif [[ -f "${WEIGHT_DIR}/seganypet_lesion.pth" ]]; then
  CKPT="${WEIGHT_DIR}/seganypet_lesion.pth"
else
  CKPT="${WEIGHT_DIR}/seganypet_v2.pth"
fi
SCRATCH_INIT=0
if [[ -z "${CKPT}" || "${CKPT}" == "none" || "${CKPT}" == "scratch" ]]; then
  SCRATCH_INIT=1
  BOARD_METHOD="${TASK1_BOARD_METHOD:-seganypet_scratch}"
  CKPT="none"
fi

DATA_ROOT="${TASK1_SEGANY_FDG_DATA:-${DATA}/task1_train_workspace/seganypet_fdg_baseline1}"
SPLITS_JSON="${CTRL}/ICLR2026/data/splits_baseline1_fdg_nnunet.json"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"

STAMP_TZ="${TASK1_STAMP_TZ:-Asia/Shanghai}"
if [[ -n "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME}"
elif [[ "${BOARD_METHOD}" == "seganypet_scratch" || "${SCRATCH_INIT}" -eq 1 ]]; then
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_seganypet_scratch_fdg_pretrain_gpu013"
else
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_seganypet_fdg_pretrain_gpu013"
fi
OUT_DIR="${REPO}/runs/${STAMP}/seganypet_fdg"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_BASE="${DATA}"

if [[ "${_ngpu}" -eq 1 ]]; then
  BS_NOTE="per-GPU 1card gpu${GPU_DEVICES}"
else
  BS_NOTE="global DP ${GPU_DEVICES}"
fi
echo "[seganypet-fdg] STAMP=${STAMP} ckpt=${CKPT} scratch=${SCRATCH_INIT} ep=${EPOCHS} bs=${BATCH_SIZE} gpus=${GPU_DEVICES} inner_cvd=${INNER_CVD} workers=${WORKERS} (${WORKERS_PER_GPU}/GPU)"

python3 "${CTRL}/ICLR2026/scripts/prepare_seganypet_fdg.py" \
  --splits-json "${SPLITS_JSON}" \
  --out-root "${DATA_ROOT}"

export TASK1_PREFLIGHT_GPUS="${PREFLIGHT_GPUS}"
export TASK1_PREFLIGHT_LABEL="iclr2026-seganypet-fdg"
bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"fdg_pretrain\":{\"status\":\"running\",\"stamp\":\"${STAMP}\",\"init\":\"${CKPT}\",\"bs\":${BATCH_SIZE},\"bs_note\":\"${BS_NOTE}\"}}},\"updated_note\":\"${BOARD_METHOD} fdg start bs=${BATCH_SIZE} ${BS_NOTE}\"}" || true

LOG="${LOG_DIR}/nohup_seganypet_fdg_${STAMP}.log"
CNAME="seganypet_fdg_${STAMP}"

# Prefer requested BS; on OOM fall back (6→3) unless disabled
BS_CANDIDATES=("${BATCH_SIZE}")
if [[ "${ALLOW_OOM_FALLBACK}" != "1" && "${BATCH_SIZE}" -gt 3 ]]; then
  BS_CANDIDATES+=(3)
fi
BS_CANDIDATES=($(printf '%s\n' "${BS_CANDIDATES[@]}" | awk '!a[$0]++'))

RC=1
USED_BS=""
for bs in "${BS_CANDIDATES[@]}"; do
  docker rm -f "${CNAME}" >/dev/null 2>&1 || true
  echo "[seganypet-fdg] try bs=${bs} → ${LOG}"
  set +e
  # background so we can arm crash-monitor while training
  nohup docker run --rm \
    --name "${CNAME}" \
    --gpus '"device='"${GPU_DEVICES}"'"' \
    -e CUDA_VISIBLE_DEVICES="${INNER_CVD}" \
    -e PYTHONPATH="${SEG_PIP}:${SEG_CODE}" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "${CTRL}:${CTRL}" \
    -v "${DATA}:${DATA}" \
    -w "${SEG_CODE}" \
    --shm-size=16g \
    "${IMAGE}" \
    python3 "${CTRL}/ICLR2026/scripts/seganypet_fewshot_finetune.py" \
      --data-root "${DATA_ROOT}" \
      --checkpoint "${CKPT}" \
      --out-dir "${OUT_DIR}" \
      --epochs "${EPOCHS}" \
      --batch-size "${bs}" \
      --accumulation-steps "${ACCUM}" \
      --lr-mode official \
      --lr "${LR}" \
      --milestones 60,85 \
      --click-max 21 \
      --num-workers "${WORKERS}" \
      --val-interval "${VAL_INTERVAL}" \
      --val-clicks "${VAL_CLICKS}" \
      --val-max-cases "${VAL_MAX_CASES}" \
      --fresh \
    >"${LOG}" 2>&1 &
  echo $! > "${OUT_DIR}/nohup.pid"
  sleep 8
  bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true
  wait "$(cat "${OUT_DIR}/nohup.pid")"
  RC=$?
  set -e
  if [[ "${RC}" -eq 0 ]]; then
    USED_BS="${bs}"
    break
  fi
  if grep -qiE 'OutOfMemoryError|CUDA out of memory' "${LOG}"; then
    echo "[seganypet-fdg] OOM at bs=${bs}; trying smaller…" >&2
    rm -f "${OUT_DIR}/latest.pth" "${OUT_DIR}/best.pth"
    continue
  fi
  echo "[error] seganypet fdg rc=${RC} (non-OOM)" >&2
  break
done

if [[ -n "${USED_BS}" ]]; then
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" \
    --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"fdg_pretrain\":{\"bs\":${USED_BS}}}},\"updated_note\":\"${BOARD_METHOD} fdg using bs=${USED_BS}\"}" || true
fi

# 阶段训练结束后保持/刷新 arm，供续训 guard 在进入 fewshot 前窗口监控
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true

BEST="${OUT_DIR}/best.pth"
[[ -f "${BEST}" ]] || BEST="${OUT_DIR}/latest.pth"
if [[ "${RC}" -eq 0 && -f "${BEST}" ]]; then
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" \
    --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"fdg_pretrain\":{\"status\":\"done\",\"stamp\":\"${STAMP}\",\"best_ckpt\":\"${BEST}\",\"bs\":${USED_BS:-${BATCH_SIZE}}}}},\"updated_note\":\"${BOARD_METHOD} fdg done\"}" || true
else
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" \
    --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"fdg_pretrain\":{\"status\":\"failed\",\"stamp\":\"${STAMP}\"}}},\"updated_note\":\"${BOARD_METHOD} fdg failed\"}" || true
fi

echo "STAMP=${STAMP}" > "${LOG_DIR}/iclr2026_seganypet_fdg_${STAMP}.txt"
echo "BEST=${BEST}" >> "${LOG_DIR}/iclr2026_seganypet_fdg_${STAMP}.txt"
echo "BS=${USED_BS:-${BATCH_SIZE}}" >> "${LOG_DIR}/iclr2026_seganypet_fdg_${STAMP}.txt"
echo "[seganypet-fdg] done rc=${RC} best=${BEST} bs=${USED_BS:-${BATCH_SIZE}}"
exit "${RC}"
