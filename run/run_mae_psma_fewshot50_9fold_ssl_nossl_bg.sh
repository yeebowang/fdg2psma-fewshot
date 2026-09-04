#!/usr/bin/env bash
# ICLR2026 · PSMA fewshot=50 × 9 covering folds
# Conditions: ssl (init=SSL continued MAE) + nossl (init=FDG MAE)
# Shared val = PSMA 10% (59). Sequential on GPU 0,1,3.
#
# Crash-monitor: DISARM during run; ARM after each fold job completes;
# DISARM before starting the next fold.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"

FT_EPOCHS="${TASK1_MAE_NUM_EPOCHS:-100}"
BATCH_SIZE="${TASK1_MAE_BATCH_SIZE:-6}"
GPUS="${TASK1_CUDA_VISIBLE_DEVICES:-0,1,3}"
DOCKER_GPUS="${TASK1_DOCKER_GPUS:-device=${GPUS}}"
PREFLIGHT_GPUS="${TASK1_PREFLIGHT_GPUS:-0 1 3}"
WORKERS_TRAIN="${TASK1_MAE_TRAIN_WORKERS:-8}"
LATE_DUAL_EPOCHS="${TASK1_MAE_LATE_DUAL_EPOCHS:-20}"
SEED="${TASK1_MAE_FEWSHOT_SEED:-42}"
N_FOLDS="${TASK1_MAE_FEWSHOT_FOLDS:-9}"
N_SHOT="${TASK1_MAE_FEWSHOT_SHOT:-50}"
# ssl | nossl | both
CONDS="${TASK1_MAE_FEWSHOT_CONDS:-both}"
# both 时调度：alternate=按折交替 ssl→nossl→ssl…；by_cond=先跑完所有 ssl 再 nossl
SCHEDULE="${TASK1_MAE_FEWSHOT_SCHEDULE:-alternate}"

SSL_STAMP_DEFAULT="20260813_051050_iclr2026_mae_psma_ssl_fewshot10_gpu013_bs6_ssl50_ft100"
SSL_STAMP="${TASK1_MAE_SSL_STAMP:-${SSL_STAMP_DEFAULT}}"
SSL_CKPT="${TASK1_MAE_SSL_CKPT:-${REPO}/runs/${SSL_STAMP}/ssl_continued/swin_mae_psma_continued_latest.pth}"
FDG_MAE_CKPT="${TASK1_MAE_FDG_CKPT:-${REPO}/weights/swinv2base/swin_mae_best_v2.pth}"

PSMA_CACHE="${DATA}/task1_train_workspace/mae_cache/psma_baseline2_70_10"
SPLIT_DIR="${CTRL}/ICLR2026/data/splits_mae_psma_fewshot50_9fold"
LOG_DIR="${CTRL}/ICLR2026/vis"

STAMP_TZ="${TASK1_STAMP_TZ:-Asia/Shanghai}"
if [[ -n "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME}"
else
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_mae_psma_fewshot${N_SHOT}_${N_FOLDS}fold_gpu013_bs${BATCH_SIZE}_${FT_EPOCHS}ep"
fi

OUT_ROOT="${REPO}/runs/${STAMP}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${PSMA_CACHE}"

export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_BASE="${DATA}"

echo "[fs50-9fold] STAMP=${STAMP}"
echo "[fs50-9fold] ssl_ckpt=${SSL_CKPT}"
echo "[fs50-9fold] fdg_mae=${FDG_MAE_CKPT}"
echo "[fs50-9fold] folds=${N_FOLDS} shot=${N_SHOT} epochs=${FT_EPOCHS} conds=${CONDS}"

[[ -f "${SSL_CKPT}" ]] || { echo "[error] missing SSL ckpt: ${SSL_CKPT}" >&2; exit 1; }
[[ -f "${FDG_MAE_CKPT}" ]] || { echo "[error] missing FDG MAE: ${FDG_MAE_CKPT}" >&2; exit 1; }

# export covering splits
python3 "${CTRL}/ICLR2026/scripts/export_mae_psma_fewshot50_9fold.py" \
  --out-dir "${SPLIT_DIR}" \
  --n-shot "${N_SHOT}" \
  --n-folds "${N_FOLDS}" \
  --seed "${SEED}"

# ensure train/val cache exists (70/10 already cached; skip ok)
bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true

# start resume guard (require_arm default)
if [[ -f "${CTRL}/run_task1_train_auto_resume_guard_bg.sh" ]]; then
  bash "${CTRL}/run_task1_train_auto_resume_guard_bg.sh" || true
fi

case "${CONDS}" in
  ssl) COND_LIST=(ssl) ;;
  nossl) COND_LIST=(nossl) ;;
  both) COND_LIST=(ssl nossl) ;;
  *) echo "[error] bad CONDS=${CONDS} (ssl|nossl|both)" >&2; exit 2 ;;
esac

run_one() {
  local cond="$1" fold="$2"
  local splits="${SPLIT_DIR}/fold${fold}_nnunet.json"
  local out_dir="${OUT_ROOT}/${cond}/fold${fold}"
  local stem="seg_psma_fs${N_SHOT}_${cond}_f${fold}"
  local png="${LOG_DIR}/loss_curve_iclr2026_mae_psma_fs${N_SHOT}_${cond}_fold${fold}_${STAMP}.png"
  local log="${LOG_DIR}/nohup_mae_psma_fs${N_SHOT}_${cond}_fold${fold}_${STAMP}.log"
  local cname="mae_fs${N_SHOT}_${cond}_f${fold}_${STAMP}"
  local foundation
  local title
  if [[ "${cond}" == "ssl" ]]; then
    foundation="${SSL_CKPT}"
    title="MAE PSMA fewshot${N_SHOT} fold${fold} (SSL)"
  else
    foundation="${FDG_MAE_CKPT}"
    title="MAE PSMA fewshot${N_SHOT} fold${fold} (noSSL)"
  fi

  mkdir -p "${out_dir}"
  echo "[fs50-9fold] === ${cond} fold${fold} ==="
  echo "[fs50-9fold] out=${out_dir}"
  echo "[fs50-9fold] log=${log}"

  bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

  set +e
  docker run --rm \
    --name "${cname}" \
    --gpus "\"${DOCKER_GPUS}\"" \
    -e CUDA_VISIBLE_DEVICES=0,1,2 \
    -v "${CTRL}:${CTRL}" \
    -v "${DATA}:${DATA}" \
    -w "${REPO}" \
    --shm-size=16g \
    "${IMAGE}" \
    python3 "${CTRL}/ICLR2026/scripts/mae_finetune_fdg_swinbase.py" \
      --cache-dir "${PSMA_CACHE}" \
      --splits-json "${splits}" \
      --foundation-ckpt "${foundation}" \
      --out-dir "${out_dir}" \
      --epochs "${FT_EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --sw-batch-size 2 \
      --val-interval 20 \
      --num-workers "${WORKERS_TRAIN}" \
      --cross-val-json "" \
      --psma-val-json "" \
      --late-dual-epochs "${LATE_DUAL_EPOCHS}" \
      --title-tag "${title}" \
      --ckpt-stem "${stem}" \
      --loss-png "${png}" \
      --fresh \
    >"${log}" 2>&1
  local rc=$?
  set -e

  bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true
  if [[ "${rc}" -ne 0 ]]; then
    echo "[error] ${cond} fold${fold} failed rc=${rc}; see ${log}" >&2
    exit "${rc}"
  fi
  echo "[fs50-9fold] ${cond} fold${fold} done"
}

job_done() {
  local cond="$1" fold="$2"
  local metrics="${OUT_ROOT}/${cond}/fold${fold}/metrics.jsonl"
  local log="${LOG_DIR}/nohup_mae_psma_fs${N_SHOT}_${cond}_fold${fold}_${STAMP}.log"
  [[ -f "${metrics}" ]] || return 1
  local n
  n="$(wc -l < "${metrics}" | tr -d ' ')"
  [[ "${n}" -ge "${FT_EPOCHS}" ]] || return 1
  if [[ -f "${log}" ]] && grep -q '\[train\] done' "${log}"; then
    return 0
  fi
  # metrics full is enough
  return 0
}

STATUS="${OUT_ROOT}/status.jsonl"
mkdir -p "${OUT_ROOT}"
if [[ ! -f "${STATUS}" ]]; then
  : >"${STATUS}"
fi
echo "{\"event\":\"start\",\"stamp\":\"${STAMP}\",\"conds\":\"${CONDS}\",\"schedule\":\"${SCHEDULE}\",\"n_folds\":${N_FOLDS}}" >>"${STATUS}"

JOBS=()
if [[ "${CONDS}" == "both" && "${SCHEDULE}" == "alternate" ]]; then
  # 按折交替：ssl f0 → nossl f0 → ssl f1 → nossl f1 → ...
  for ((fold=0; fold<N_FOLDS; fold++)); do
    for cond in ssl nossl; do
      JOBS+=("${cond}:${fold}")
    done
  done
elif [[ "${CONDS}" == "both" ]]; then
  for cond in ssl nossl; do
    for ((fold=0; fold<N_FOLDS; fold++)); do
      JOBS+=("${cond}:${fold}")
    done
  done
else
  for ((fold=0; fold<N_FOLDS; fold++)); do
    JOBS+=("${CONDS}:${fold}")
  done
fi

echo "[fs50-9fold] schedule=${SCHEDULE} jobs=${#JOBS[@]}"

for job in "${JOBS[@]}"; do
  cond="${job%%:*}"
  fold="${job##*:}"
  if job_done "${cond}" "${fold}"; then
    echo "[fs50-9fold] skip done ${cond} fold${fold}"
    echo "{\"event\":\"fold_skip\",\"cond\":\"${cond}\",\"fold\":${fold}}" >>"${STATUS}"
    continue
  fi
  t0=$(date +%s)
  run_one "${cond}" "${fold}"
  t1=$(date +%s)
  echo "{\"event\":\"fold_done\",\"cond\":\"${cond}\",\"fold\":${fold},\"sec\":$((t1-t0))}" >>"${STATUS}"
done

# aggregate val best dice
python3 - <<PY
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
conds = ["ssl", "nossl"] if "${CONDS}" == "both" else ["${CONDS}"]
summary = {"stamp": "${STAMP}", "n_folds": ${N_FOLDS}, "n_shot": ${N_SHOT}, "conds": {}}
for cond in conds:
    dices = []
    for fold in range(${N_FOLDS}):
        m = root / cond / f"fold{fold}" / "metrics.jsonl"
        best = None
        if m.is_file():
            for line in m.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                vd = r.get("val_dice")
                if vd is not None and vd == vd:
                    if best is None or vd > best:
                        best = float(vd)
        dices.append(best)
    ok = [d for d in dices if d is not None]
    mean = (sum(ok) / len(ok)) if ok else None
    std = (sum((x - mean) ** 2 for x in ok) / len(ok)) ** 0.5 if ok and mean is not None else None
    summary["conds"][cond] = {
        "fold_best_dice": dices,
        "mean": mean,
        "std": std,
        "n_ok": len(ok),
    }
out = root / "aggregate_val_dice.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
vis = Path("${LOG_DIR}") / f"aggregate_mae_psma_fs${N_SHOT}_9fold_{STAMP}.json"
vis.write_text(json.dumps(summary, indent=2) + "\n")
print(f"[fs50-9fold] wrote {out}")
print(f"[fs50-9fold] wrote {vis}")
PY

echo "[fs50-9fold] ALL DONE STAMP=${STAMP}"
echo "STAMP=${STAMP}" > "${LOG_DIR}/iclr2026_mae_psma_fewshot${N_SHOT}_9fold_${STAMP}.txt"
echo "OUT_ROOT=${OUT_ROOT}" >> "${LOG_DIR}/iclr2026_mae_psma_fewshot${N_SHOT}_9fold_${STAMP}.txt"
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true
