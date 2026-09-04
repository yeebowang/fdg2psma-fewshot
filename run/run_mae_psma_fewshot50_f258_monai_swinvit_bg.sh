#!/usr/bin/env bash
# Fewshot50 on folds 2,5,8 ONLY — labeled finetune from MONAI generic SwinViT SSL
# (Tang et al. model_swinvit.pt), NOT FDG MAE / continued SSL.
#
# Parallel: f2→GPU0, f5→GPU1, f8→GPU3 · single-GPU bs=2
# Arch: depths=2,2,2,2 · use_v2=0 (native to monai_swinvit)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"

FT_EPOCHS="${TASK1_MAE_NUM_EPOCHS:-100}"
BATCH_SIZE="${TASK1_MAE_BATCH_SIZE:-2}"
BB_LR_MULT="${TASK1_MAE_BACKBONE_LR_MULT:-0.1}"
FREEZE_ENC_EP="${TASK1_MAE_FREEZE_ENCODER_EPOCHS:-20}"
FOLDS_CSV="${TASK1_MAE_FEWSHOT_FOLDS_CSV:-2,5,8}"
FT_GPU_LIST="${TASK1_MAE_FT_GPU_LIST:-0 1 3}"
WORKERS_TRAIN="${TASK1_MAE_TRAIN_WORKERS:-4}"
LATE_DUAL_EPOCHS="${TASK1_MAE_LATE_DUAL_EPOCHS:-20}"

FOUNDATION="${TASK1_MAE_GENERIC_CKPT:-${REPO}/weights/generic/model_swinvit.pt}"
FOUNDATION_KIND="${TASK1_MAE_FOUNDATION_KIND:-monai_swinvit}"
DEPTHS="${TASK1_MAE_DEPTHS:-2,2,2,2}"
USE_V2="${TASK1_MAE_USE_V2:-0}"

SPLIT_DIR="${CTRL}/ICLR2026/data/splits_mae_psma_fewshot50_9fold"
PSMA_CACHE="${DATA}/task1_train_workspace/mae_cache/psma_baseline2_70_10"
LOG_DIR="${CTRL}/ICLR2026/vis"

STAMP_TZ="${TASK1_STAMP_TZ:-Asia/Shanghai}"
if [[ -n "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME}"
else
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_mae_psma_fs50_monai_swinvit_f258_gpu013"
fi

OUT_ROOT="${REPO}/runs/${STAMP}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_BASE="${DATA}"

echo "[monai-fs50] STAMP=${STAMP}"
echo "[monai-fs50] foundation=${FOUNDATION} kind=${FOUNDATION_KIND} depths=${DEPTHS} use_v2=${USE_V2}"
echo "[monai-fs50] FT ep=${FT_EPOCHS} bs=${BATCH_SIZE} bb_lr=${BB_LR_MULT} freeze=${FREEZE_ENC_EP} folds=${FOLDS_CSV}"

[[ -f "${FOUNDATION}" ]] || { echo "[error] missing ${FOUNDATION}" >&2; exit 1; }

python3 "${CTRL}/ICLR2026/scripts/export_mae_psma_fewshot50_9fold.py" \
  --out-dir "${SPLIT_DIR}" --n-shot 50 --n-folds 9 --seed 42

bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true

IFS=',' read -r -a FOLD_ARR <<< "${FOLDS_CSV}"
read -r -a GPU_ARR <<< "${FT_GPU_LIST}"
[[ "${#GPU_ARR[@]}" -ge "${#FOLD_ARR[@]}" ]] || {
  echo "[error] need >=${#FOLD_ARR[@]} GPUs" >&2
  exit 1
}

STATUS="${OUT_ROOT}/status.jsonl"
echo "{\"event\":\"start\",\"stamp\":\"${STAMP}\",\"folds\":\"${FOLDS_CSV}\",\"foundation\":\"${FOUNDATION_KIND}\"}" >>"${STATUS}"

run_one_bg() {
  local fold="$1" gpu="$2"
  local splits="${SPLIT_DIR}/fold${fold}_nnunet.json"
  local out_dir="${OUT_ROOT}/monai/fold${fold}"
  local stem="seg_psma_fs50_monai_f${fold}"
  local png="${LOG_DIR}/loss_curve_iclr2026_mae_psma_fs50_monai_fold${fold}_${STAMP}.png"
  local log="${LOG_DIR}/nohup_mae_psma_fs50_monai_fold${fold}_${STAMP}.log"
  local cname="mae_fs50_monai_f${fold}_${STAMP}"
  local fresh_flag=(--fresh)
  if [[ "${TASK1_MAE_FT_FORCE_FRESH:-0}" != "1" ]] && [[ -f "${out_dir}/latest_${stem}.pth" ]]; then
    fresh_flag=()
    echo "[monai-fs50] resume f${fold}"
  fi
  mkdir -p "${out_dir}"
  docker rm -f "${cname}" >/dev/null 2>&1 || true
  echo "[monai-fs50] === fold${fold} → GPU${gpu} ==="
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
      --title-tag "MAE PSMA fs50 fold${fold} (MONAI SwinViT SSL)" \
      --ckpt-stem "${stem}" \
      --loss-png "${png}" \
      "${fresh_flag[@]}" \
    >"${log}" 2>&1 &
  echo $! >"${out_dir}/docker_host_pid.txt"
}

bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
t0=$(date +%s)
for i in "${!FOLD_ARR[@]}"; do
  run_one_bg "${FOLD_ARR[$i]}" "${GPU_ARR[$i]}"
done

any_fail=0
for i in "${!FOLD_ARR[@]}"; do
  fold="${FOLD_ARR[$i]}"
  out_dir="${OUT_ROOT}/monai/fold${fold}"
  pid="$(cat "${out_dir}/docker_host_pid.txt")"
  set +e
  wait "${pid}"
  rc=$?
  set -e
  echo "{\"event\":\"fold_done\",\"fold\":${fold},\"rc\":${rc}}" >>"${STATUS}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "[error] monai f${fold} rc=${rc} log=${LOG_DIR}/nohup_mae_psma_fs50_monai_fold${fold}_${STAMP}.log" >&2
    any_fail=1
  fi
done
t1=$(date +%s)
echo "{\"event\":\"wave_done\",\"sec\":$((t1-t0))}" >>"${STATUS}"
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true
[[ "${any_fail}" -eq 0 ]] || exit 1

python3 - <<PY
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
folds = [int(x) for x in "${FOLDS_CSV}".split(",") if x.strip()]
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
    "protocol": "fewshot50_monai_swinvit",
    "foundation": "${FOUNDATION}",
    "foundation_kind": "${FOUNDATION_KIND}",
    "depths": "${DEPTHS}",
    "use_v2": int("${USE_V2}"),
    "fold_best_dice": dices,
    "mean": mean,
    "std": std,
    "n_ok": len(ok),
}
out = root / "aggregate_val_dice_f258.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
vis = Path("${LOG_DIR}") / f"aggregate_mae_psma_fs50_monai_f258_{STAMP}.json"
vis.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "[monai-fs50] ALL DONE STAMP=${STAMP}"
echo "STAMP=${STAMP}" > "${LOG_DIR}/iclr2026_mae_psma_monai_f258_${STAMP}.txt"
