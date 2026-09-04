#!/usr/bin/env bash
# MONAI SwinViT · PSMA fewshot50 f2/5/8 from FDG supervised seg
# Protocol: FDG bs=6 · PSMA fewshot bs=2
# Default: try parallel 1GPU/fold; on OOM → sequential DP on 0,1,3 (still bs=2)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"

FEWSHOT_N="${TASK1_FEWSHOT_N:-50}"
BOARD_STAGE="${TASK1_PSMA_BOARD_STAGE:-psma_fs${FEWSHOT_N}_f258}"

FT_EPOCHS="${TASK1_MAE_NUM_EPOCHS:-100}"
BATCH_SIZE="${TASK1_MAE_BATCH_SIZE:-2}"
BB_LR_MULT="${TASK1_MAE_BACKBONE_LR_MULT:-1.0}"
FREEZE_ENC_EP="${TASK1_MAE_FREEZE_ENCODER_EPOCHS:-0}"
FOLDS_CSV="${TASK1_MAE_FEWSHOT_FOLDS_CSV:-2,5,8}"
FT_GPU_LIST="${TASK1_MAE_FT_GPU_LIST:-0 1 3}"
SEQ_GPUS="${TASK1_MAE_SEQ_GPUS:-0,1,3}"
FORCE_SEQ="${TASK1_MAE_F258_FORCE_SEQ:-0}"
WORKERS_TRAIN="${TASK1_MAE_TRAIN_WORKERS:-4}"
LATE_DUAL_EPOCHS="${TASK1_MAE_LATE_DUAL_EPOCHS:-20}"

FOUNDATION="${TASK1_MONAI_FDG_SEG_CKPT:-}"
[[ -n "${FOUNDATION}" ]] || { echo "[error] set TASK1_MONAI_FDG_SEG_CKPT to best_seg_fdg_monai.pth" >&2; exit 1; }
FOUNDATION_KIND="${TASK1_MAE_FOUNDATION_KIND:-seg}"
BOARD_METHOD="${TASK1_BOARD_METHOD:-monai_swinvit}"
DEPTHS="${TASK1_MAE_DEPTHS:-2,2,2,2}"
USE_V2="${TASK1_MAE_USE_V2:-0}"

SPLIT_DIR="${TASK1_FEWSHOT_SPLIT_DIR:-${CTRL}/ICLR2026/data/splits_mae_psma_fewshot${FEWSHOT_N}_9fold}"
PSMA_CACHE="${DATA}/task1_train_workspace/mae_cache/psma_baseline2_70_10"
LOG_DIR="${CTRL}/ICLR2026/vis"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${LOG_DIR}/iclr2026_aligned_fdg_fs50_f258_board.json}"

STAMP_TZ="${TASK1_STAMP_TZ:-Asia/Shanghai}"
if [[ -n "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME}"
elif [[ "${BOARD_METHOD}" == "monai_scratch" ]]; then
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_monai_scratch_psma_fs${FEWSHOT_N}_from_fdg_seg_f258_gpu013"
else
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_monai_psma_fs${FEWSHOT_N}_from_fdg_seg_f258_gpu013"
fi

OUT_ROOT="${REPO}/runs/${STAMP}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_BASE="${DATA}"

echo "[monai-fdgseg-fs${FEWSHOT_N}] STAMP=${STAMP} foundation=${FOUNDATION} bs=${BATCH_SIZE} force_seq=${FORCE_SEQ}"
[[ -f "${FOUNDATION}" ]] || { echo "[error] missing ${FOUNDATION}" >&2; exit 1; }

[[ -f "${SPLIT_DIR}/fold0_nnunet.json" ]] || python3 "${CTRL}/ICLR2026/scripts/export_mae_psma_fewshot50_9fold.py" \
  --out-dir "${SPLIT_DIR}" --n-shot "${FEWSHOT_N}" --n-folds 9 --seed 42
bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true

IFS=',' read -r -a FOLD_ARR <<< "${FOLDS_CSV}"
read -r -a GPU_ARR <<< "${FT_GPU_LIST}"
STATUS="${OUT_ROOT}/status.jsonl"
MODE_FILE="${OUT_ROOT}/f258_mode.txt"
echo "parallel" >"${MODE_FILE}"

_log_of() { echo "${LOG_DIR}/nohup_monai_psma_fs${FEWSHOT_N}_fdgseg_fold${1}_${STAMP}.log"; }
_is_oom_log() { [[ -f "$1" ]] && grep -qiE 'OutOfMemoryError|CUDA out of memory' "$1"; }
_clear_fold() {
  local fold="$1" out_dir="${OUT_ROOT}/monai/fold${fold}" stem="seg_psma_fs${FEWSHOT_N}_monai_fdgseg_f${fold}"
  rm -f "${out_dir}/latest_${stem}.pth" "${out_dir}/best_${stem}.pth" \
        "${out_dir}/metrics.jsonl" "${out_dir}/docker_host_pid.txt" || true
}
_stop_all() {
  local fold
  for fold in "${FOLD_ARR[@]}"; do
    docker rm -f "monai_fs${FEWSHOT_N}_fdgseg_f${fold}_${STAMP}" >/dev/null 2>&1 || true
  done
}

run_one_parallel_bg() {
  local fold="$1" gpu="$2"
  local splits="${SPLIT_DIR}/fold${fold}_nnunet.json"
  local out_dir="${OUT_ROOT}/monai/fold${fold}"
  local stem="seg_psma_fs${FEWSHOT_N}_monai_fdgseg_f${fold}"
  local png="${LOG_DIR}/loss_curve_iclr2026_monai_psma_fs${FEWSHOT_N}_fdgseg_fold${fold}_${STAMP}.png"
  local log; log="$(_log_of "${fold}")"
  local cname="monai_fs${FEWSHOT_N}_fdgseg_f${fold}_${STAMP}"
  mkdir -p "${out_dir}"
  docker rm -f "${cname}" >/dev/null 2>&1 || true
  echo "[monai-fdgseg-fs${FEWSHOT_N}] PARALLEL fold${fold} → GPU${gpu} bs=${BATCH_SIZE}"
  nohup docker run --rm \
    --name "${cname}" \
    --gpus "device=${gpu}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "${CTRL}:${CTRL}" \
    -v "${DATA}:${DATA}" \
    -w "${REPO}" \
    --shm-size=8g \
    "${IMAGE}" \
    python3 "${CTRL}/ICLR2026/scripts/mae_finetune_fdg_swinbase.py" \
      --cache-dir "${PSMA_CACHE}" \
      --splits-json "${splits}" \
      --foundation-ckpt "${FOUNDATION}" \
      --foundation-kind "${FOUNDATION_KIND}" \
      --depths "${DEPTHS}" \
      --use-v2 "${USE_V2}" \
      --out-dir "${out_dir}" \
      --epochs "${FT_EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --sw-batch-size 2 \
      --val-interval 20 \
      --num-workers "${WORKERS_TRAIN}" \
      --backbone-lr-mult "${BB_LR_MULT}" \
      --freeze-encoder-epochs "${FREEZE_ENC_EP}" \
      --cross-val-json "" \
      --psma-val-json "" \
      --late-dual-epochs "${LATE_DUAL_EPOCHS}" \
      --title-tag "MONAI PSMA fs50 fold${fold} bs${BATCH_SIZE} parallel" \
      --ckpt-stem "${stem}" \
      --loss-png "${png}" \
      --fresh \
    >"${log}" 2>&1 &
  echo $! >"${out_dir}/docker_host_pid.txt"
}

run_one_sequential_fg() {
  local fold="$1"
  local splits="${SPLIT_DIR}/fold${fold}_nnunet.json"
  local out_dir="${OUT_ROOT}/monai/fold${fold}"
  local stem="seg_psma_fs${FEWSHOT_N}_monai_fdgseg_f${fold}"
  local png="${LOG_DIR}/loss_curve_iclr2026_monai_psma_fs${FEWSHOT_N}_fdgseg_fold${fold}_${STAMP}.png"
  local log; log="$(_log_of "${fold}")"
  local cname="monai_fs${FEWSHOT_N}_fdgseg_f${fold}_${STAMP}"
  mkdir -p "${out_dir}"
  docker rm -f "${cname}" >/dev/null 2>&1 || true
  echo "[monai-fdgseg-fs${FEWSHOT_N}] SEQUENTIAL fold${fold} → GPUs ${SEQ_GPUS} bs=${BATCH_SIZE} (DP)"
  docker run --rm \
    --name "${cname}" \
    --gpus '"device='"${SEQ_GPUS}"'"' \
    -e CUDA_VISIBLE_DEVICES=0,1,2 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "${CTRL}:${CTRL}" \
    -v "${DATA}:${DATA}" \
    -w "${REPO}" \
    --shm-size=16g \
    "${IMAGE}" \
    python3 "${CTRL}/ICLR2026/scripts/mae_finetune_fdg_swinbase.py" \
      --cache-dir "${PSMA_CACHE}" \
      --splits-json "${splits}" \
      --foundation-ckpt "${FOUNDATION}" \
      --foundation-kind "${FOUNDATION_KIND}" \
      --depths "${DEPTHS}" \
      --use-v2 "${USE_V2}" \
      --out-dir "${out_dir}" \
      --epochs "${FT_EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --sw-batch-size 2 \
      --val-interval 20 \
      --num-workers "${WORKERS_TRAIN}" \
      --backbone-lr-mult "${BB_LR_MULT}" \
      --freeze-encoder-epochs "${FREEZE_ENC_EP}" \
      --cross-val-json "" \
      --psma-val-json "" \
      --late-dual-epochs "${LATE_DUAL_EPOCHS}" \
      --title-tag "MONAI PSMA fs50 fold${fold} bs${BATCH_SIZE} sequential-DP" \
      --ckpt-stem "${stem}" \
      --loss-png "${png}" \
      --fresh \
    >"${log}" 2>&1
}

bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

MODE="parallel"
BS_NOTE="per-GPU parallel"
[[ "${FORCE_SEQ}" == "1" ]] && { MODE="sequential"; BS_NOTE="seq DP ${SEQ_GPUS}"; }

python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
  --board "${BOARD_JSON}" \
  --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"${BOARD_STAGE}\":{\"status\":\"running\",\"stamp\":\"${STAMP}\",\"foundation\":\"${FOUNDATION}\",\"bs\":${BATCH_SIZE},\"bs_note\":\"${BS_NOTE}\",\"total_epochs\":${FT_EPOCHS}}}},\"updated_note\":\"${BOARD_METHOD} fewshot bs=${BATCH_SIZE} mode=${MODE}\"}" || true

any_fail=0
had_oom=0

if [[ "${MODE}" == "parallel" ]]; then
  n_gpu="${#GPU_ARR[@]}"
  [[ "${n_gpu}" -ge 1 ]] || { echo "[error] empty GPU list" >&2; exit 1; }
  idx=0
  n_fold="${#FOLD_ARR[@]}"
  while [[ "${idx}" -lt "${n_fold}" ]]; do
    wave_end=$((idx + n_gpu))
    [[ "${wave_end}" -gt "${n_fold}" ]] && wave_end="${n_fold}"
    echo "[monai-fdgseg-fs${FEWSHOT_N}] parallel wave folds ${FOLD_ARR[*]:idx:$((wave_end - idx))} on GPUs ${GPU_ARR[*]}"
    for ((i = idx; i < wave_end; i++)); do
      g=$((i - idx))
      run_one_parallel_bg "${FOLD_ARR[$i]}" "${GPU_ARR[$g]}"
    done
    for ((i = idx; i < wave_end; i++)); do
      fold="${FOLD_ARR[$i]}"
      set +e
      wait "$(cat "${OUT_ROOT}/monai/fold${fold}/docker_host_pid.txt")"
      rc=$?
      set -e
      echo "{\"event\":\"fold_done\",\"mode\":\"parallel\",\"fold\":${fold},\"rc\":${rc}}" >>"${STATUS}"
      if [[ "${rc}" -ne 0 ]]; then
        any_fail=1
        _is_oom_log "$(_log_of "${fold}")" && had_oom=1
      fi
    done
    idx="${wave_end}"
  done
  if [[ "${had_oom}" -eq 1 ]]; then
    echo "[monai-fdgseg-fs${FEWSHOT_N}] OOM under parallel → SEQUENTIAL fallback, bs=${BATCH_SIZE}"
    _stop_all
    for fold in "${FOLD_ARR[@]}"; do _clear_fold "${fold}"; done
    echo "sequential" >"${MODE_FILE}"
    MODE="sequential"
    BS_NOTE="seq DP ${SEQ_GPUS} (OOM fallback)"
    any_fail=0
    python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
      --board "${BOARD_JSON}" \
      --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"${BOARD_STAGE}\":{\"bs\":${BATCH_SIZE},\"bs_note\":\"${BS_NOTE}\"}}},\"updated_note\":\"${BOARD_METHOD} OOM→sequential bs=${BATCH_SIZE}\"}" || true
  fi
fi

if [[ "${MODE}" == "sequential" ]]; then
  for fold in "${FOLD_ARR[@]}"; do
    set +e
    run_one_sequential_fg "${fold}"
    rc=$?
    set -e
    echo "{\"event\":\"fold_done\",\"mode\":\"sequential\",\"fold\":${fold},\"rc\":${rc}}" >>"${STATUS}"
    if [[ "${rc}" -ne 0 ]]; then
      any_fail=1
      break
    fi
  done
fi

bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true

python3 - <<PY
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
folds = [int(x) for x in "${FOLDS_CSV}".split(",") if x.strip()]
mode = Path("${MODE_FILE}").read_text().strip() if Path("${MODE_FILE}").is_file() else "unknown"
dices = []
for fold in folds:
    m = root / "monai" / f"fold{fold}" / "metrics.jsonl"
    best = None
    if m.is_file():
        for line in m.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            vd = r.get("val_dice")
            if vd is not None and vd == vd:
                best = float(vd) if best is None or float(vd) > best else best
    dices.append(best)
ok = [d for d in dices if d is not None]
mean = sum(ok) / len(ok) if ok else None
std = (sum((x - mean) ** 2 for x in ok) / len(ok)) ** 0.5 if ok and mean is not None else None
summary = {
    "stamp": "${STAMP}",
    "folds": folds,
    "protocol": "fewshot50_monai_from_fdg_supervised_seg",
    "foundation": "${FOUNDATION}",
    "batch_size": int("${BATCH_SIZE}"),
    "f258_mode": mode,
    "fold_best_dice": {str(f): d for f, d in zip(folds, dices)},
    "mean": mean,
    "std": std,
    "n_ok": len(ok),
}
(root / "aggregate_val_dice_f258.json").write_text(json.dumps(summary, indent=2) + "\n")
Path("${LOG_DIR}", f"aggregate_monai_psma_fs${FEWSHOT_N}_fdgseg_f258_${STAMP}.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
PY

if [[ "${any_fail}" -eq 0 ]]; then
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" --ingest-monai-stamp "${STAMP}" || true
else
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" \
    --patch-json "{\"methods\":{\"${BOARD_METHOD}\":{\"${BOARD_STAGE}\":{\"status\":\"failed\",\"stamp\":\"${STAMP}\"}}},\"updated_note\":\"${BOARD_METHOD} fewshot failed\"}" || true
fi
[[ "${any_fail}" -eq 0 ]] || exit 1
echo "[monai-fdgseg-fs${FEWSHOT_N}] ALL DONE ${STAMP} mode=$(cat "${MODE_FILE}")"
