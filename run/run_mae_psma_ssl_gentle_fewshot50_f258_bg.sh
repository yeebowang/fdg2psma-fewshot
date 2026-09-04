#!/usr/bin/env bash
# Revised SSL (gentle + freeze early Swin) → fewshot50 on folds 2,5,8 only
# Conditions: ssl (new SSL best) vs nossl (FDG MAE), wave schedule.
# Fewshot: backbone_lr_mult=0.1, freeze_encoder_epochs=20
# Parallel: each fold on one GPU (f2→GPU0, f5→GPU1, f8→GPU3); ssl wave then nossl wave.
# Skip Stage A if SSL best already exists: TASK1_MAE_SKIP_SSL=1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
REPO="${CTRL}/ICLR2026/3D-MAE-PET-CT"
IMAGE="${TASK1_MAE_IMAGE:-iclr2026_3dmae_petct:cu118}"

FT_EPOCHS="${TASK1_MAE_NUM_EPOCHS:-100}"
# single-GPU parallel: bs=6 OOMs at encoder-unfreeze (ep21); keep local≈旧三卡 DP 每卡 2
BATCH_SIZE="${TASK1_MAE_BATCH_SIZE:-2}"
SSL_EPOCHS="${TASK1_MAE_SSL_EPOCHS:-20}"
SSL_BATCH_SIZE="${TASK1_MAE_SSL_BATCH_SIZE:-3}"
SSL_LR="${TASK1_MAE_SSL_LR:-3e-5}"
ALIGN_W="${TASK1_MAE_ALIGN_WEIGHT:-0.05}"
FREEZE_STAGES="${TASK1_MAE_SSL_FREEZE_STAGES:-2}"
BB_LR_MULT="${TASK1_MAE_BACKBONE_LR_MULT:-0.1}"
FREEZE_ENC_EP="${TASK1_MAE_FREEZE_ENCODER_EPOCHS:-20}"
FOLDS_CSV="${TASK1_MAE_FEWSHOT_FOLDS_CSV:-2,5,8}"
SKIP_SSL="${TASK1_MAE_SKIP_SSL:-0}"

GPUS="${TASK1_CUDA_VISIBLE_DEVICES:-0,1,3}"
DOCKER_GPUS="${TASK1_DOCKER_GPUS:-device=${GPUS}}"
PREFLIGHT_GPUS="${TASK1_PREFLIGHT_GPUS:-0 1 3}"
# fewer workers when 3 folds share the host
WORKERS_TRAIN="${TASK1_MAE_TRAIN_WORKERS:-4}"
LATE_DUAL_EPOCHS="${TASK1_MAE_LATE_DUAL_EPOCHS:-20}"
# space-separated physical GPU ids aligned with FOLDS_CSV order
FT_GPU_LIST="${TASK1_MAE_FT_GPU_LIST:-0 1 3}"

FDG_MAE_CKPT="${TASK1_MAE_FDG_CKPT:-${REPO}/weights/swinv2base/swin_mae_best_v2.pth}"
PSMA_SPLITS="${CTRL}/ICLR2026/data/splits_baseline2_psma_uda_nnunet.json"
FDG_SPLITS="${CTRL}/ICLR2026/data/splits_baseline1_fdg_nnunet.json"
SPLIT_DIR="${CTRL}/ICLR2026/data/splits_mae_psma_fewshot50_9fold"
PSMA_CACHE="${DATA}/task1_train_workspace/mae_cache/psma_baseline2_70_10"
FDG_CACHE="${DATA}/task1_train_workspace/mae_cache/fdg_baseline1_70_10"
LOG_DIR="${CTRL}/ICLR2026/vis"

STAMP_TZ="${TASK1_STAMP_TZ:-Asia/Shanghai}"
if [[ -n "${TASK1_NNUNET_RESULTS_STAMP_NAME:-}" ]]; then
  STAMP="${TASK1_NNUNET_RESULTS_STAMP_NAME}"
else
  STAMP="$(TZ="${STAMP_TZ}" date +%Y%m%d_%H%M%S)_iclr2026_mae_psma_ssl_gentle_fs50_f258_gpu013"
fi

OUT_ROOT="${REPO}/runs/${STAMP}"
OUT_SSL="${OUT_ROOT}/ssl_continued_gentle"
SSL_BEST="${OUT_SSL}/swin_mae_psma_continued_best.pth"
SSL_LATEST="${OUT_SSL}/swin_mae_psma_continued_latest.pth"
SSL_PNG="${LOG_DIR}/loss_curve_iclr2026_mae_psma_ssl_gentle_${STAMP}.png"
SSL_LOG="${LOG_DIR}/nohup_mae_psma_ssl_gentle_${STAMP}.log"

mkdir -p "${OUT_SSL}" "${LOG_DIR}"
export TASK1_NNUNET_RESULTS_STAMP_NAME="${STAMP}"
export TASK1_BASE="${DATA}"

echo "[gentle-f258] STAMP=${STAMP}"
echo "[gentle-f258] SSL: ep=${SSL_EPOCHS} lr=${SSL_LR} align=${ALIGN_W} freeze_stages=${FREEZE_STAGES} skip=${SKIP_SSL}"
echo "[gentle-f258] FT: ep=${FT_EPOCHS} bb_lr_mult=${BB_LR_MULT} freeze_enc=${FREEZE_ENC_EP} folds=${FOLDS_CSV}"
echo "[gentle-f258] FT parallel GPUs: ${FT_GPU_LIST} (1 fold / GPU)"

[[ -f "${FDG_MAE_CKPT}" ]] || { echo "[error] missing FDG MAE" >&2; exit 1; }

# ensure fewshot50 splits exist
python3 "${CTRL}/ICLR2026/scripts/export_mae_psma_fewshot50_9fold.py" \
  --out-dir "${SPLIT_DIR}" --n-shot 50 --n-folds 9 --seed 42

bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true

# ---------- Stage A: gentle SSL ----------
if [[ "${SKIP_SSL}" == "1" && -f "${SSL_BEST}" ]]; then
  echo "[gentle-f258] Stage A SKIPPED (existing ${SSL_BEST})"
elif [[ "${SKIP_SSL}" == "1" ]]; then
  echo "[error] TASK1_MAE_SKIP_SSL=1 but missing ${SSL_BEST}" >&2
  exit 1
else
  bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
  echo "[gentle-f258] Stage A gentle SSL → ${SSL_LOG}"
  set +e
  docker run --rm \
    --name "mae_ssl_gentle_${STAMP}" \
    --gpus "\"${DOCKER_GPUS}\"" \
    -e CUDA_VISIBLE_DEVICES=0,1,2 \
    -v "${CTRL}:${CTRL}" \
    -v "${DATA}:${DATA}" \
    -w "${REPO}" \
    --shm-size=16g \
    "${IMAGE}" \
    python3 "${CTRL}/ICLR2026/scripts/mae_continued_ssl_psma.py" \
      --fdg-mae-ckpt "${FDG_MAE_CKPT}" \
      --psma-splits-json "${PSMA_SPLITS}" \
      --fdg-splits-json "${FDG_SPLITS}" \
      --psma-cache-dir "${PSMA_CACHE}" \
      --fdg-cache-dir "${FDG_CACHE}" \
      --out-dir "${OUT_SSL}" \
      --epochs "${SSL_EPOCHS}" \
      --batch-size "${SSL_BATCH_SIZE}" \
      --lr "${SSL_LR}" \
      --align-weight "${ALIGN_W}" \
      --freeze-swin-stages "${FREEZE_STAGES}" \
      --num-workers "${WORKERS_TRAIN}" \
      --loss-png "${SSL_PNG}" \
      --fresh \
    >"${SSL_LOG}" 2>&1
  SSL_RC=$?
  set -e
  bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true
  [[ "${SSL_RC}" -eq 0 ]] || { echo "[error] SSL failed rc=${SSL_RC}" >&2; exit "${SSL_RC}"; }
fi
[[ -f "${SSL_BEST}" ]] || SSL_BEST="${SSL_LATEST}"
[[ -f "${SSL_BEST}" ]] || { echo "[error] missing SSL ckpt" >&2; exit 1; }
echo "[gentle-f258] SSL ckpt=${SSL_BEST}"

# ---------- Stage B: fewshot folds (parallel per wave) ----------
IFS=',' read -r -a FOLD_ARR <<< "${FOLDS_CSV}"
read -r -a GPU_ARR <<< "${FT_GPU_LIST}"
if [[ "${#GPU_ARR[@]}" -lt "${#FOLD_ARR[@]}" ]]; then
  echo "[error] need >=${#FOLD_ARR[@]} GPUs in FT_GPU_LIST, got ${#GPU_ARR[@]}" >&2
  exit 1
fi
STATUS="${OUT_ROOT}/status.jsonl"
mkdir -p "${OUT_ROOT}"
echo "{\"event\":\"start\",\"stamp\":\"${STAMP}\",\"folds\":\"${FOLDS_CSV}\",\"ssl\":\"gentle\",\"parallel\":1,\"gpus\":\"${FT_GPU_LIST}\"}" >>"${STATUS}"

run_one_bg() {
  local cond="$1" fold="$2" gpu="$3"
  local splits="${SPLIT_DIR}/fold${fold}_nnunet.json"
  local out_dir="${OUT_ROOT}/${cond}/fold${fold}"
  local stem="seg_psma_fs50_${cond}_f${fold}_gentle"
  local png="${LOG_DIR}/loss_curve_iclr2026_mae_psma_fs50_${cond}_fold${fold}_gentle_${STAMP}.png"
  local log="${LOG_DIR}/nohup_mae_psma_fs50_${cond}_fold${fold}_gentle_${STAMP}.log"
  local cname="mae_fs50_${cond}_f${fold}_g_${STAMP}"
  local foundation title fresh_flag
  if [[ "${cond}" == "ssl" ]]; then
    foundation="${SSL_BEST}"
    title="MAE PSMA fs50 fold${fold} (gentleSSL)"
  else
    foundation="${FDG_MAE_CKPT}"
    title="MAE PSMA fs50 fold${fold} (noSSL+bbLR)"
  fi
  # auto-resume if latest exists (unless force fresh)
  fresh_flag=(--fresh)
  if [[ "${TASK1_MAE_FT_FORCE_FRESH:-0}" != "1" ]] && [[ -f "${out_dir}/latest_${stem}.pth" ]]; then
    fresh_flag=()
    echo "[gentle-f258] resume ${cond} f${fold} from latest"
  fi
  mkdir -p "${out_dir}"
  # drop stale container name
  docker rm -f "${cname}" >/dev/null 2>&1 || true
  echo "[gentle-f258] === ${cond} fold${fold} → GPU${gpu} ==="
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
      --foundation-ckpt "${foundation}" \
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
      --title-tag "${title}" \
      --ckpt-stem "${stem}" \
      --loss-png "${png}" \
      "${fresh_flag[@]}" \
    >"${log}" 2>&1 &
  echo $! >"${out_dir}/docker_host_pid.txt"
}

wait_wave() {
  local cond="$1"
  local -a pids=()
  local -a folds_wait=()
  local i fold out_dir pid cname rc any_fail=0
  for i in "${!FOLD_ARR[@]}"; do
    fold="${FOLD_ARR[$i]}"
    out_dir="${OUT_ROOT}/${cond}/fold${fold}"
    cname="mae_fs50_${cond}_f${fold}_g_${STAMP}"
    if [[ -f "${out_dir}/docker_host_pid.txt" ]]; then
      pid="$(cat "${out_dir}/docker_host_pid.txt")"
      pids+=("${pid}")
      folds_wait+=("${fold}")
    fi
  done
  echo "[gentle-f258] waiting ${cond} wave pids=${pids[*]:-none}"
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    fold="${folds_wait[$i]}"
    set +e
    wait "${pid}"
    rc=$?
    set -e
    echo "{\"event\":\"fold_done\",\"cond\":\"${cond}\",\"fold\":${fold},\"rc\":${rc}}" >>"${STATUS}"
    if [[ "${rc}" -ne 0 ]]; then
      echo "[error] ${cond} f${fold} rc=${rc} log=${LOG_DIR}/nohup_mae_psma_fs50_${cond}_fold${fold}_gentle_${STAMP}.log" >&2
      any_fail=1
    fi
  done
  [[ "${any_fail}" -eq 0 ]] || exit 1
}

# ssl wave then nossl wave: all folds in parallel
bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true
for cond in ssl nossl; do
  t0=$(date +%s)
  for i in "${!FOLD_ARR[@]}"; do
    run_one_bg "${cond}" "${FOLD_ARR[$i]}" "${GPU_ARR[$i]}"
  done
  wait_wave "${cond}"
  t1=$(date +%s)
  echo "{\"event\":\"wave_done\",\"cond\":\"${cond}\",\"sec\":$((t1-t0))}" >>"${STATUS}"
done
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true

python3 - <<PY
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
folds = [int(x) for x in "${FOLDS_CSV}".split(",") if x.strip()!=""]
summary = {"stamp": "${STAMP}", "folds": folds, "protocol": "gentle_ssl+bbLR", "conds": {}}
for cond in ["ssl", "nossl"]:
    dices = []
    for fold in folds:
        m = root / cond / f"fold{fold}" / "metrics.jsonl"
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
    mean = sum(ok)/len(ok) if ok else None
    std = (sum((x-mean)**2 for x in ok)/len(ok))**0.5 if ok and mean is not None else None
    summary["conds"][cond] = {"fold_best_dice": dices, "mean": mean, "std": std, "n_ok": len(ok)}
out = root / "aggregate_val_dice_f258.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
vis = Path("${LOG_DIR}") / f"aggregate_mae_psma_fs50_f258_gentle_{STAMP}.json"
vis.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "[gentle-f258] ALL DONE STAMP=${STAMP}"
echo "STAMP=${STAMP}" > "${LOG_DIR}/iclr2026_mae_psma_gentle_f258_${STAMP}.txt"
bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true
