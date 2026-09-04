#!/usr/bin/env bash
# SegAnyPET fewshot50 f2/5/8 — OFFICIAL hyperparams (train_cpcl defaults):
#   epochs=200, batch_size=12, accumulation=20, lr=8e-4 (encoder), prompt/decoder x0.1,
#   milestones=120,180, click_max=21
#
# If bs=12 cannot fit on 1 GPU (expected), each fold runs SEQUENTIALLY and
# monopolizes GPUs 0,1,3 via DataParallel (global bs=12 → 4/GPU).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"
SEG_CODE="${CTRL}/ICLR2026/third_party/SegAnyPET/code"
SEG_PIP="${CTRL}/ICLR2026/third_party/seganypet_pip"
WEIGHT_DIR="${REPO}/weights/seganypet"

FT_EPOCHS="${TASK1_SEGANY_OFFICIAL_EPOCHS:-200}"
BATCH_SIZE="${TASK1_SEGANY_OFFICIAL_BATCH_SIZE:-12}"
ACCUM="${TASK1_SEGANY_OFFICIAL_ACCUM:-20}"
FOLDS_CSV="${TASK1_SEGANY_FOLDS_CSV:-2,5,8}"
GPU_DEVICES="${TASK1_SEGANY_OFFICIAL_GPUS:-0,1,3}"
WORKERS="${TASK1_SEGANY_WORKERS:-4}"
VAL_INTERVAL="${TASK1_SEGANY_VAL_INTERVAL:-20}"
VAL_CLICKS="${TASK1_SEGANY_VAL_CLICKS:-5}"
VAL_MAX_CASES="${TASK1_SEGANY_VAL_MAX_CASES:-15}"
LR="${TASK1_SEGANY_OFFICIAL_LR:-8e-4}"

if [[ -n "${TASK1_SEGANY_CKPT:-}" ]]; then
  CKPT="${TASK1_SEGANY_CKPT}"
elif [[ -f "${WEIGHT_DIR}/seganypet_lesion.pth" ]]; then
  CKPT="${WEIGHT_DIR}/seganypet_lesion.pth"
elif [[ -f "${WEIGHT_DIR}/seganypet_v2.pth" ]]; then
  CKPT="${WEIGHT_DIR}/seganypet_v2.pth"
else
  echo "[error] missing SegAnyPET weights under ${WEIGHT_DIR}" >&2
  exit 1
fi

DATA_ROOT="${TASK1_SEGANY_DATA_ROOT:-${DATA}/task1_train_workspace/seganypet_fewshot50_f258}"
SPLIT_DIR="${CTRL}/ICLR2026/data/splits_mae_psma_fewshot50_9fold"
LOG_DIR="${CTRL}/ICLR2026/vis"

STAMP_TZ="${TASK1_STAMP_TZ:-Asia/Shanghai}"
if [[ -n "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME}"
else
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_seganypet_official_fs50_f258_gpu013"
fi

OUT_ROOT="${REPO}/runs/${STAMP}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_BASE="${DATA}"

echo "[seganypet-official] STAMP=${STAMP}"
echo "[seganypet-official] ckpt=${CKPT}"
echo "[seganypet-official] ep=${FT_EPOCHS} bs=${BATCH_SIZE} accum=${ACCUM} lr=${LR} gpus=${GPU_DEVICES} folds=${FOLDS_CSV}"

[[ -d "${SEG_PIP}/torchio" ]] || {
  echo "[error] torchio not in ${SEG_PIP}" >&2
  exit 1
}

python3 "${CTRL}/ICLR2026/scripts/prepare_seganypet_fewshot_f258.py" \
  --folds "${FOLDS_CSV}" \
  --split-dir "${SPLIT_DIR}" \
  --out-root "${DATA_ROOT}"

bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true
# disarm may require stamp; tolerate missing
TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}" bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

IFS=',' read -r -a FOLD_ARR <<< "${FOLDS_CSV}"
STATUS="${OUT_ROOT}/status.jsonl"
echo "{\"event\":\"start\",\"stamp\":\"${STAMP}\",\"protocol\":\"official\",\"ckpt\":\"${CKPT}\",\"folds\":\"${FOLDS_CSV}\",\"bs\":${BATCH_SIZE},\"accum\":${ACCUM}}" >>"${STATUS}"

# Try batch sizes: prefer requested; on OOM fall back
BS_CANDIDATES=("${BATCH_SIZE}")
if [[ "${BATCH_SIZE}" -gt 6 ]]; then
  BS_CANDIDATES+=(6)
fi
if [[ "${BATCH_SIZE}" -gt 3 ]]; then
  BS_CANDIDATES+=(3)
fi
# unique preserve order
BS_CANDIDATES=($(printf '%s\n' "${BS_CANDIDATES[@]}" | awk '!a[$0]++'))
run_fold() {
  local fold="$1"
  local fold_data="${DATA_ROOT}/fold${fold}"
  local out_dir="${OUT_ROOT}/seganypet/fold${fold}"
  local log="${LOG_DIR}/nohup_seganypet_official_fold${fold}_${STAMP}.log"
  local cname="seganypet_official_f${fold}_${STAMP}"
  local fresh_flag=(--fresh)
  if [[ "${TASK1_SEGANY_FORCE_FRESH:-0}" != "1" ]] && [[ -f "${out_dir}/latest.pth" ]]; then
    fresh_flag=()
    echo "[seganypet-official] resume f${fold}"
  fi
  mkdir -p "${out_dir}"

  local ok=0
  local used_bs=""
  for bs in "${BS_CANDIDATES[@]}"; do
    docker rm -f "${cname}" >/dev/null 2>&1 || true
    # clear OOM-broken fresh start only when switching bs mid-attempt without latest
    echo "[seganypet-official] === fold${fold} → GPUs ${GPU_DEVICES} bs=${bs} (try) ==="
    set +e
    docker run --rm \
      --name "${cname}" \
      --gpus '"device='"${GPU_DEVICES}"'"' \
      -e CUDA_VISIBLE_DEVICES=0,1,2 \
      -e PYTHONPATH="${SEG_PIP}:${SEG_CODE}" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -v "${CTRL}:${CTRL}" \
      -v "${DATA}:${DATA}" \
      -w "${SEG_CODE}" \
      --shm-size=16g \
      "${IMAGE}" \
      python3 "${CTRL}/ICLR2026/scripts/seganypet_fewshot_finetune.py" \
        --data-root "${fold_data}" \
        --checkpoint "${CKPT}" \
        --out-dir "${out_dir}" \
        --epochs "${FT_EPOCHS}" \
        --batch-size "${bs}" \
        --accumulation-steps "${ACCUM}" \
        --lr-mode official \
        --lr "${LR}" \
        --milestones 120,180 \
        --click-max 21 \
        --num-workers "${WORKERS}" \
        --val-interval "${VAL_INTERVAL}" \
        --val-clicks "${VAL_CLICKS}" \
        --val-max-cases "${VAL_MAX_CASES}" \
        "${fresh_flag[@]}" \
      >"${log}" 2>&1
    local rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      ok=1
      used_bs="${bs}"
      break
    fi
    if grep -qiE 'OutOfMemoryError|CUDA out of memory' "${log}"; then
      echo "[seganypet-official] f${fold} OOM at bs=${bs}; trying smaller…" >&2
      # force fresh on next bs try (broken first steps)
      fresh_flag=(--fresh)
      rm -f "${out_dir}/latest.pth"
      continue
    fi
    echo "[error] seganypet official f${fold} rc=${rc} (non-OOM) log=${log}" >&2
    return "${rc}"
  done
  [[ "${ok}" -eq 1 ]] || return 1
  echo "{\"event\":\"fold_done\",\"fold\":${fold},\"bs\":${used_bs},\"rc\":0}" >>"${STATUS}"
  echo "[seganypet-official] fold${fold} DONE bs=${used_bs}"
  return 0
}

t0=$(date +%s)
any_fail=0
for fold in "${FOLD_ARR[@]}"; do
  if ! run_fold "${fold}"; then
    any_fail=1
    break
  fi
done
t1=$(date +%s)
echo "{\"event\":\"wave_done\",\"sec\":$((t1 - t0))}" >>"${STATUS}"
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true
[[ "${any_fail}" -eq 0 ]] || exit 1

python3 - <<PY
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
folds = [int(x) for x in "${FOLDS_CSV}".split(",") if x.strip()]
dices = []
for fold in folds:
    best = None
    m = root / "seganypet" / f"fold{fold}" / "metrics.jsonl"
    if m.is_file():
        for line in m.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            vd = r.get("val_dice")
            if vd is not None and vd == vd:
                best = float(vd) if best is None or float(vd) > best else best
    s = root / "seganypet" / f"fold{fold}" / "summary.json"
    if s.is_file():
        bj = json.loads(s.read_text()).get("best_val_dice")
        if bj is not None and (best is None or bj > best):
            best = float(bj)
    dices.append(best)
ok = [d for d in dices if d is not None]
mean = sum(ok) / len(ok) if ok else None
std = (sum((x - mean) ** 2 for x in ok) / len(ok)) ** 0.5 if ok and mean is not None else None
summary = {
    "stamp": "${STAMP}",
    "folds": folds,
    "protocol": "fewshot50_seganypet_official_click",
    "checkpoint": "${CKPT}",
    "hparams": {
        "epochs": int("${FT_EPOCHS}"),
        "batch_size_target": int("${BATCH_SIZE}"),
        "accumulation_steps": int("${ACCUM}"),
        "lr": float("${LR}"),
        "lr_mode": "official",
        "milestones": [120, 180],
        "gpus": "${GPU_DEVICES}",
    },
    "note": "official train_cpcl hparams; 3-GPU DataParallel per fold if needed; val Dice with GT clicks",
    "fold_best_dice": dices,
    "mean": mean,
    "std": std,
    "n_ok": len(ok),
}
out = root / "aggregate_val_dice_f258.json"
out.write_text(json.dumps(summary, indent=2) + "\\n")
vis = Path("${LOG_DIR}") / f"aggregate_seganypet_official_fs50_f258_${STAMP}.json"
vis.write_text(json.dumps(summary, indent=2) + "\\n")
print(json.dumps(summary, indent=2))
PY

echo "[seganypet-official] ALL DONE STAMP=${STAMP}"
echo "STAMP=${STAMP}" > "${LOG_DIR}/iclr2026_seganypet_official_f258_${STAMP}.txt"
