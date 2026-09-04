#!/usr/bin/env bash
# ICLR2026 baseline2：FDG→PSMA UDA 伪标分割微调（1A+2B）
# 每 round：predict(421) → CC+μ/h/λ 伪标 → 训 10ep（tr70/val50）→ 取 PSMA val best 进下一 round
# R=200，λ: 0.1→0.8；初始化 = FDG checkpoint_best（FDG val_loss）
#
#   export TASK1_BASE=/media/ybwang/data1/PSMA-DATA
#   bash ICLR2026/run/run_baseline2_psma_uda_selftrain_bg.sh
#
# 续训/崩溃：外层读 uda_state.json；单 round 训中崩溃靠 guard + CONTINUE。
# 停止：TASK1_NNUNET_RESULTS_STAMP_NAME=<PARENT_STAMP> bash scripts/task1_stop_train_and_resume.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK1_REPO_ROOT="${ROOT}"
export TASK1_BASE="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
export WORK_DIR="${WORK_DIR:-${TASK1_BASE}/task1_train_workspace}"
WORK="${WORK_DIR}"
VIS="${WORK}/01_train_vis"
mkdir -p "${VIS}/log"

DATASET_ID="${DATASET_ID:-228}"
DS="Dataset${DATASET_ID}_AutoPETIV_Task1_2ch"
TRAINER="${TRAINER:-nnUNetTrainer_Task1StdTrainVal50}"
PLANS_ID="${PLANS_ID:-nnUNetPlans}"
CONFIG="${CONFIG:-3d_fullres}"
# 强制 fold 为 0..4 或 all；拒绝环境里残留的路径污染
_raw_fold="${FOLD:-0}"
if [[ "${_raw_fold}" =~ ^(all|[0-4])$ ]]; then
  FOLD="${_raw_fold}"
else
  echo "[baseline2] WARN ignore invalid FOLD=${_raw_fold}；改用 0" >&2
  FOLD=0
fi
TF="${TRAINER}__${PLANS_ID}__${CONFIG}"

ROUNDS="${TASK1_UDA_ROUNDS:-200}"
EP_PER_ROUND="${TASK1_UDA_EPOCHS_PER_ROUND:-10}"
TRAIN_ITERS="${TASK1_TRAIN_ITERS_PER_EPOCH:-70}"
VAL_ITERS="${TASK1_VAL_ITERS_PER_EPOCH:-50}"
LAMBDA_START="${TASK1_UDA_LAMBDA_START:-0.1}"
LAMBDA_END="${TASK1_UDA_LAMBDA_END:-0.8}"
PRED_EVERY="${TASK1_UDA_PRED_EVERY:-1}"  # 每 N round 重推；默认每 round

SPLITS_JSON="${TASK1_SPLITS_FINAL_JSON:-${ROOT}/ICLR2026/data/splits_baseline2_psma_uda_nnunet.json}"
[[ -f "${SPLITS_JSON}" ]] || {
  echo "[error] missing splits: ${SPLITS_JSON}" >&2
  echo "  run: python3 ICLR2026/scripts/export_baseline2_psma_uda_splits.py" >&2
  exit 1
}

PREP_DIR="${WORK}/nnUNet_preprocessed/${DS}/nnUNetPlans_${CONFIG}"
[[ -d "${PREP_DIR}" ]] || { echo "[error] missing prep ${PREP_DIR}" >&2; exit 1; }

FDG_STAMP="${TASK1_UDA_FDG_STAMP:-20260810_104431_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val10_3000ep}"
FDG_BEST="${TASK1_UDA_FDG_BEST:-${WORK}/nnUNet_results/${FDG_STAMP}/${DS}/${TF}/fold_${FOLD}/checkpoint_best.pth}"
[[ -f "${FDG_BEST}" ]] || { echo "[error] missing FDG best: ${FDG_BEST}" >&2; exit 1; }

# 父 STAMP（整次 UDA 实验）；每 round 子 STAMP = ${PARENT}__rXXX（双下划线，避免与名中 R200 冲突）
_req="${TASK1_NNUNET_RESULTS_STAMP_NAME:-}"
if [[ -n "${_req}" && "${_req}" == *iclr2026_baseline2* ]]; then
  if [[ "${_req}" =~ __r[0-9]{3}$ ]]; then
    PARENT_STAMP="${_req%__r*}"
  else
    PARENT_STAMP="${_req}"
  fi
else
  if [[ -n "${_req}" ]]; then
    echo "[baseline2] ignore non-baseline2 STAMP=${_req}；生成新 STAMP" >&2
  fi
  PARENT_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_baseline2_psma_uda_R${ROUNDS}_ep${EP_PER_ROUND}_tr${TRAIN_ITERS}_val${VAL_ITERS}"
fi

UDA_ROOT="${WORK}/nnUNet_results/${PARENT_STAMP}"
mkdir -p "${UDA_ROOT}"
STATE_JSON="${UDA_ROOT}/uda_state.json"
PRIOR_JSON="${UDA_ROOT}/psma_uda_prior_state.json"
ICLR_VIS="${TASK1_LOSS_OUT_DIR:-${ROOT}/ICLR2026/vis}"
mkdir -p "${ICLR_VIS}"

# 默认后台：整段 round 循环 nohup；调试可 TASK1_UDA_FOREGROUND=1
if [[ "${TASK1_UDA_FOREGROUND:-0}" != "1" && "${TASK1_UDA_WORKER:-0}" != "1" ]]; then
  export TASK1_UDA_WORKER=1
  export TASK1_NNUNET_RESULTS_STAMP_NAME="${PARENT_STAMP}"
  BG_LOG="${ICLR_VIS}/nohup_baseline2_${PARENT_STAMP}.log"
  nohup bash "${ROOT}/ICLR2026/run/run_baseline2_psma_uda_selftrain_bg.sh" \
    >"${BG_LOG}" 2>&1 &
  echo "[baseline2] launched bg pid=$! PARENT=${PARENT_STAMP}"
  echo "  log=${BG_LOG}"
  echo "  state=${STATE_JSON}"
  echo "  stop: TASK1_NNUNET_RESULTS_STAMP_NAME=${PARENT_STAMP} bash scripts/task1_stop_train_and_resume.sh"
  exit 0
fi

RESTART="ICLR2026/run/run_baseline2_psma_uda_selftrain_bg.sh"
IMAGE_TAG="${IMAGE_TAG:-autopet_baseline}"

_start_round_guard() {
  local round_stamp="$1"
  local extra="$2"
  TASK1_GUARD_STAMP="${round_stamp}" \
  TASK1_NNUNET_RESULTS_STAMP_NAME="${round_stamp}" \
  TASK1_GUARD_TRAINER_FOLDER="${TF}" \
  TASK1_GUARD_DATASET_DIR="${DS}" \
  TASK1_GUARD_TOTAL_EPOCHS="${EP_PER_ROUND}" \
  TASK1_GUARD_RESTART_SCRIPT="${RESTART}" \
  TASK1_GUARD_REQUIRE_ARM=1 \
  TASK1_GUARD_EXTRA_ENV="${extra},TASK1_NNUNET_RESULTS_STAMP_NAME=${PARENT_STAMP}" \
  FOLD="${FOLD}" \
    bash "${ROOT}/run_task/run_task1_train_auto_resume_guard_bg.sh" || true
}

export TASK1_STOP_AFTER_PREP=0
export TASK1_SKIP_RAW_PREP=1
export TASK1_SKIP_PLAN_PREP=1
export TASK1_FORCE_PLAN_PREP=0
export TASK1_FIXED_BATCH_3D_FULLRES="${TASK1_FIXED_BATCH_3D_FULLRES:-6}"
export TASK1_N_PROC_DA="${TASK1_N_PROC_DA:-6}"
export TASK1_SEGMENT_CHECKPOINT=0
export TASK1_PERIODIC_CHECKPOINT_EVERY=0
export TASK1_ALWAYS_SAVE_LATEST=1
export TASK1_SAVE_LATEST_EVERY=1
export TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH=0
export TASK1_RAM_SHARD_ENABLE=0
export TASK1_SPLITS_FINAL_JSON="${SPLITS_JSON}"
export TASK1_BEST_BY="${TASK1_BEST_BY:-val_loss}"
export TASK1_VAL_LOSS_ONLY="${TASK1_VAL_LOSS_ONLY:-1}"
# UDA 微调：默认 LR 从 0.01 → 1e-3，避免 1ep 内塌成全背景
export TASK1_INITIAL_LR="${TASK1_INITIAL_LR:-0.001}"
export TASK1_LOSS_PLOT_VAL_EMA=0
export TASK1_PSMA_VAL_ENABLE=0
export TRAINER CONFIG PLANS_ID FOLD DATASET_ID
export TASK1_DATASET_NAME="${DS}"
export TASK1_NNUNET_RESULTS_STAMP_SUBDIR=1
export TASK1_TRAINER_PY="${TASK1_TRAINER_PY:-${ROOT}/nnunet_ext_trainers/nnUNetTrainer_Task1StdTrainVal50.py}"
export TASK1_TRAIN_NUM_GPUS="${TASK1_TRAIN_NUM_GPUS:-3}"
export TASK1_DOCKER_GPUS="${TASK1_DOCKER_GPUS:-device=0,1,3}"
export TASK1_CUDA_VISIBLE_DEVICES="${TASK1_CUDA_VISIBLE_DEVICES:-0,1,3}"
export TASK1_PREFLIGHT_GPUS="${TASK1_PREFLIGHT_GPUS:-0 1 3}"
_nproc_da="${TASK1_N_PROC_DA}"
_shm_g=$(( _nproc_da * 3 ))
(( _shm_g < 16 )) && _shm_g=16
export TASK1_DOCKER_SHM="${TASK1_DOCKER_SHM:-${_shm_g}g}"
# UDA 外层必须同步等每 round 训完；忽略环境里残留的 BACKGROUND=1（否则 docker -d 立刻返回，误报 missing best）
export TASK1_DOCKER_BACKGROUND=0
echo "[baseline2] TASK1_DOCKER_BACKGROUND=0（强制同步等训完）"
export TASK1_LOSS_OUT_DIR="${ICLR_VIS}"
export TASK1_LOSS_OUT_DIR_EXTRA=none
# 全 200 round 合成到一张图；关闭每 round 的 plot_nnunet_loss_live，避免盖掉合成图
export TASK1_LIVE_LOSS_PLOT=0
export TASK1_LOSS_MERGE_ALL_LOGS=1
export TASK1_LOSS_PLOT_X_FOLLOW=1
export TASK1_LOSS_PLOT_SHOW_ETA="${TASK1_LOSS_PLOT_SHOW_ETA:-1}"
export TASK1_LOSS_PLOT_SHOW_ETA=1
export TASK1_LOSS_PLOT_SEED_EMPTY=0
# 合成图 x 轴钉死 0..R*ep
export TASK1_LOSS_PLOT_X_MAX_EPOCHS=""
export TASK1_LOSS_PLOT_VAL_FROM_EPOCH=""

PARENT_LOSS_PNG="${ICLR_VIS}/loss_curve_iclr2026_baseline2_${PARENT_STAMP}.png"
export TASK1_LOSS_OUT_NAME="$(basename "${PARENT_LOSS_PNG}")"

# 读/写状态
_read_start_round() {
  if [[ -f "${STATE_JSON}" ]]; then
    python3 - <<PY
import json
s=json.load(open("${STATE_JSON}"))
# next round to run; if phase=train incomplete, re-run same round
r=int(s.get("next_round", s.get("round", 0)))
phase=str(s.get("phase", "predict"))
print(r if phase != "done" else ${ROUNDS})
PY
  else
    echo 0
  fi
}

_write_state() {
  local r="$1" phase="$2" best="${3:-}" round_stamp="${4:-}"
  python3 - <<PY
import json
from pathlib import Path
p=Path("${STATE_JSON}")
s={}
if p.is_file():
    s=json.loads(p.read_text())
s.update({
  "parent_stamp": "${PARENT_STAMP}",
  "round": int("${r}"),
  "next_round": int("${r}") + (1 if "${phase}" == "round_done" else 0),
  "phase": "${phase}",
  "best_ckpt": "${best}",
  "round_stamp": "${round_stamp}",
  "rounds_total": int("${ROUNDS}"),
  "epochs_per_round": int("${EP_PER_ROUND}"),
})
if "${phase}" == "round_done":
    s["next_round"] = int("${r}") + 1
    s["phase"] = "predict"
p.write_text(json.dumps(s, indent=2) + "\n")
print(f"[baseline2] state → {p} round={s['round']} phase={s['phase']} next={s['next_round']}")
PY
}

START_R="$(_read_start_round)"
echo "[baseline2] PARENT=${PARENT_STAMP} start_round=${START_R} R=${ROUNDS} ep/round=${EP_PER_ROUND} tr=${TRAIN_ITERS} val=${VAL_ITERS}"
echo "[baseline2] FDG_BEST=${FDG_BEST}"
echo "[baseline2] splits=${SPLITS_JSON}"
echo "[baseline2] combined loss PNG → ${PARENT_LOSS_PNG}"

# 整次实验一张合成 loss 图（推理 ETA + 跨 round 拼接）
docker rm -f "uda_combined_plotter_${PARENT_STAMP}" 2>/dev/null || true
mkdir -p "${ICLR_VIS}" "${UDA_ROOT}"
touch "${PARENT_LOSS_PNG}" 2>/dev/null || true
chmod a+rw "${PARENT_LOSS_PNG}" 2>/dev/null || true
docker run -d --rm \
  --name "uda_combined_plotter_${PARENT_STAMP}" \
  -v "${ROOT}:${ROOT}" \
  -v "${WORK}:${WORK}" \
  -v "${TASK1_BASE}:${TASK1_BASE}" \
  -w "${ROOT}" \
  -e MPLCONFIGDIR=/tmp/mpl \
  --entrypoint python3 \
  "${IMAGE_TAG}" \
  ICLR2026/scripts/psma_uda_combined_loss_plot.py \
    --parent-stamp "${PARENT_STAMP}" \
    --work "${WORK}" \
    --out-png "${PARENT_LOSS_PNG}" \
    --state-json "${STATE_JSON}" \
    --status-json "${UDA_ROOT}/uda_combined_plot_status.json" \
    --rounds-total "${ROUNDS}" \
    --epochs-per-round "${EP_PER_ROUND}" \
    --train-iters "${TRAIN_ITERS}" \
    --val-iters "${VAL_ITERS}" \
    --dataset-name "${DS}" \
    --trainer-folder "${TF}" \
    --fold "${FOLD}" \
    --poll 15 \
  >/dev/null
echo "[baseline2] combined loss plotter → ${PARENT_LOSS_PNG}"

# 崩溃续跑：重拉本脚本即可（读 uda_state.json）。各 round 训中另启 ROUND stamp 的 guard。
EXTRA_ENV="TASK1_UDA_WORKER=1,TASK1_UDA_FOREGROUND=1,TASK1_UDA_ROUNDS=${ROUNDS},TASK1_UDA_EPOCHS_PER_ROUND=${EP_PER_ROUND},TASK1_TRAIN_ITERS_PER_EPOCH=${TRAIN_ITERS},TASK1_VAL_ITERS_PER_EPOCH=${VAL_ITERS},TASK1_FIXED_BATCH_3D_FULLRES=${TASK1_FIXED_BATCH_3D_FULLRES},TASK1_N_PROC_DA=${TASK1_N_PROC_DA},TASK1_SPLITS_FINAL_JSON=${SPLITS_JSON},TASK1_BEST_BY=${TASK1_BEST_BY},TASK1_VAL_LOSS_ONLY=${TASK1_VAL_LOSS_ONLY},TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH=0,TASK1_NNUNET_RESULTS_STAMP_NAME=${PARENT_STAMP},TASK1_NNUNET_RESULTS_STAMP_SUBDIR=1,DATASET_ID=${DATASET_ID},TASK1_DATASET_NAME=${DS},CONFIG=${CONFIG},FOLD=${FOLD},TRAINER=${TRAINER},PLANS_ID=${PLANS_ID},TASK1_SKIP_RAW_PREP=1,TASK1_SKIP_PLAN_PREP=1,TASK1_STOP_AFTER_PREP=0,TASK1_TRAIN_NUM_GPUS=${TASK1_TRAIN_NUM_GPUS},TASK1_DOCKER_GPUS=${TASK1_DOCKER_GPUS},TASK1_CUDA_VISIBLE_DEVICES=${TASK1_CUDA_VISIBLE_DEVICES},TASK1_UDA_FDG_BEST=${FDG_BEST},TASK1_UDA_FDG_STAMP=${FDG_STAMP},TASK1_LOSS_OUT_DIR=${ICLR_VIS},TASK1_LIVE_LOSS_PLOT=0,TASK1_LOSS_PLOT_SHOW_ETA=1,TASK1_LOSS_PLOT_X_FOLLOW=1,TASK1_LOSS_PLOT_VAL_EMA=0,TASK1_DOCKER_BACKGROUND=0,TASK1_INITIAL_LR=${TASK1_INITIAL_LR}"

CUR_BEST="${FDG_BEST}"
CUR_RESULTS="$(dirname "$(dirname "$(dirname "$(dirname "${FDG_BEST}")")")")"  # .../STAMP

if [[ -f "${STATE_JSON}" ]]; then
  _prev_best="$(python3 -c "import json; print(json.load(open('${STATE_JSON}')).get('best_ckpt','') or '')")"
  if [[ -n "${_prev_best}" && -f "${_prev_best}" ]]; then
    CUR_BEST="${_prev_best}"
    CUR_RESULTS="$(dirname "$(dirname "$(dirname "$(dirname "${CUR_BEST}")")")")"
  fi
fi

for (( r=START_R; r<ROUNDS; r++ )); do
  ROUND_STAMP="$(printf '%s__r%03d' "${PARENT_STAMP}" "${r}")"
  ROUND_DIR="${WORK}/nnUNet_results/${ROUND_STAMP}/${DS}/${TF}/fold_${FOLD}"
  ROUND_WORK="${UDA_ROOT}/round_$(printf '%03d' "${r}")"
  PRED_OUT="${ROUND_WORK}/pred"
  LABELS_DIR="${ROUND_WORK}/labelsTr_pseudo"
  SEG_B2ND="${ROUND_WORK}/pseudo_seg_b2nd"
  mkdir -p "${PRED_OUT}/pred" "${LABELS_DIR}" "${SEG_B2ND}"
  chmod -R a+rwX "${ROUND_WORK}" "${UDA_ROOT}" 2>/dev/null || true

  echo ""
  echo "========== baseline2 round ${r}/${ROUNDS} =========="
  _write_state "${r}" "predict" "${CUR_BEST}" "${ROUND_STAMP}"

  TASK1_CRASH_MONITOR_STAGE="baseline2_r${r}_before_predict" \
  TASK1_NNUNET_RESULTS_STAMP_NAME="${PARENT_STAMP}" \
    bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true

  NEED_PRED=1
  if (( PRED_EVERY > 1 )) && (( r % PRED_EVERY != 0 )) && [[ -d "${PRED_OUT}/pred" ]]; then
    NEED_PRED=0
    echo "[baseline2] skip predict (PRED_EVERY=${PRED_EVERY})"
  fi

  mkdir -p "${PRED_OUT}/pred" "${ICLR_VIS}"
  chmod a+rwX "${PRED_OUT}" "${ICLR_VIS}" 2>/dev/null || true
  date +%s >"${PRED_OUT}/predict_t0.txt"
  chmod a+rw "${PRED_OUT}/predict_t0.txt" 2>/dev/null || true
  # 合成图由 uda_combined_plotter 轮询 uda_state + pred；不再每 round 新开 PNG

  if [[ "${NEED_PRED}" == "1" ]]; then
    export TASK1_UDA_CKPT="${CUR_BEST}"
    export TASK1_UDA_PRED_OUT="${PRED_OUT}"
    export TASK1_UDA_CASES_JSON="${SPLITS_JSON}"
    export TASK1_UDA_NNUNET_RESULTS="${CUR_RESULTS}"
    bash "${ROOT}/ICLR2026/scripts/psma_uda_predict_train.sh"
  fi

  _write_state "${r}" "pseudo" "${CUR_BEST}" "${ROUND_STAMP}"
  if ! docker run --rm \
    -v "${ROOT}:${ROOT}" \
    -v "${WORK}:${WORK}" \
    -v "${TASK1_BASE}:${TASK1_BASE}" \
    -w "${ROOT}" \
    --entrypoint python3 \
    "${IMAGE_TAG}" \
    ICLR2026/scripts/psma_uda_make_pseudo_labels.py \
      --pred-dir "${PRED_OUT}/pred" \
      --cases-json "${SPLITS_JSON}" \
      --out-labels-dir "${LABELS_DIR}" \
      --prior-state "${PRIOR_JSON}" \
      --round "${r}" \
      --rounds-total "${ROUNDS}" \
      --lambda-start "${LAMBDA_START}" \
      --lambda-end "${LAMBDA_END}"; then
    echo "[error] round ${r} pseudo failed（可能模型塌缩/空标）；保留 CUR_BEST=${CUR_BEST} 并中止" >&2
    exit 2
  fi

  docker run --rm \
    -v "${ROOT}:${ROOT}" \
    -v "${WORK}:${WORK}" \
    -v "${TASK1_BASE}:${TASK1_BASE}" \
    -w "${ROOT}" \
    --entrypoint python3 \
    "${IMAGE_TAG}" \
    ICLR2026/scripts/psma_uda_pseudo_nii_to_seg_b2nd.py \
      --prep-dir "${PREP_DIR}" \
      --labels-dir "${LABELS_DIR}" \
      --out-seg-b2nd-dir "${SEG_B2ND}" \
      --cases-json "${SPLITS_JSON}"

  _write_state "${r}" "train" "${CUR_BEST}" "${ROUND_STAMP}"

  # 本 round 训练：新 stamp，从 CUR_BEST 作 pretrained；10 ep
  export TASK1_NNUNET_RESULTS_STAMP_NAME="${ROUND_STAMP}"
  export TASK1_NUM_EPOCHS="${EP_PER_ROUND}"
  export TASK1_TRAIN_ITERS_PER_EPOCH="${TRAIN_ITERS}"
  export TASK1_VAL_ITERS_PER_EPOCH="${VAL_ITERS}"
  export TASK1_PRETRAINED_WEIGHTS="${CUR_BEST}"
  export TASK1_CONTINUE_TRAINING=0
  export TASK1_CONTINUE_FROM_LATEST=0
  export TASK1_CONTINUE_FROM_BEST=0
  export TASK1_CONTINUE_PICK_NEWER=0
  export TASK1_PSEUDO_SEG_B2ND_DIR="${SEG_B2ND}"
  export TASK1_LOSS_OUT_NAME="$(basename "${PARENT_LOSS_PNG}")"
  export TASK1_LIVE_LOSS_PLOT=0
  export TASK1_LR_SCHEDULE_NUM_EPOCHS="${EP_PER_ROUND}"
  export TASK1_INITIAL_LR="${TASK1_INITIAL_LR:-0.001}"

  # 若本 round 已有未完成 checkpoint → 改为 continue
  if [[ -f "${ROUND_DIR}/checkpoint_latest.pth" || -f "${ROUND_DIR}/checkpoint_final.pth" || -f "${ROUND_DIR}/checkpoint_best.pth" ]]; then
    _done="$(docker run --rm -v "${WORK}:${WORK}" --entrypoint python3 "${IMAGE_TAG}" -c "
from pathlib import Path
import torch
fold=Path('${ROUND_DIR}')
for name in ('checkpoint_final.pth','checkpoint_latest.pth','checkpoint_best.pth'):
    p=fold/name
    if p.is_file():
        try:
            ck=torch.load(str(p), map_location='cpu', weights_only=False)
        except TypeError:
            ck=torch.load(str(p), map_location='cpu')
        print(int(ck.get('current_epoch', -1)))
        break
else:
    print(-1)
")"
    if [[ "${_done}" -ge $((EP_PER_ROUND - 1)) ]] && [[ -f "${ROUND_DIR}/checkpoint_best.pth" ]]; then
      echo "[baseline2] round ${r} already finished (ep=${_done}); reuse best"
    else
      echo "[baseline2] resume round ${r} train from ep≈${_done}"
      export TASK1_PRETRAINED_WEIGHTS=""
      export TASK1_CONTINUE_TRAINING=1
      export TASK1_CONTINUE_FROM_LATEST=1
      _start_round_guard "${ROUND_STAMP}" "${EXTRA_ENV},TASK1_CONTINUE_TRAINING=1,TASK1_CONTINUE_FROM_LATEST=1,TASK1_PSEUDO_SEG_B2ND_DIR=${SEG_B2ND},TASK1_NUM_EPOCHS=${EP_PER_ROUND}"
      TASK1_CRASH_MONITOR_STAGE="baseline2_r${r}_train_resume" \
      TASK1_NNUNET_RESULTS_STAMP_NAME="${ROUND_STAMP}" \
        bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
      export TASK1_PREFLIGHT_STEP="iclr2026-baseline2-r${r}-resume"
      bash "${ROOT}/scripts/task1_gpu_train_preflight.sh" || true
      sleep 2
      TASK1_CRASH_MONITOR_STAGE="baseline2_r${r}_train_running" \
      TASK1_NNUNET_RESULTS_STAMP_NAME="${ROUND_STAMP}" \
      TASK1_CRASH_MONITOR_ARM_SEC="${TASK1_CRASH_MONITOR_ARM_SEC:-86400}" \
        bash "${ROOT}/scripts/task1_crash_monitor_arm.sh" || true
      export TASK1_DOCKER_BACKGROUND=0
      bash "${ROOT}/other/task1_train_nnunet_from_dataset1.sh"
      TASK1_CRASH_MONITOR_STAGE="baseline2_r${r}_train_done" \
      TASK1_NNUNET_RESULTS_STAMP_NAME="${ROUND_STAMP}" \
        bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
    fi
  else
    _start_round_guard "${ROUND_STAMP}" "${EXTRA_ENV},TASK1_PRETRAINED_WEIGHTS=${CUR_BEST},TASK1_PSEUDO_SEG_B2ND_DIR=${SEG_B2ND},TASK1_NUM_EPOCHS=${EP_PER_ROUND},TASK1_CONTINUE_TRAINING=0,TASK1_DOCKER_BACKGROUND=0"
    TASK1_CRASH_MONITOR_STAGE="baseline2_r${r}_train_start" \
    TASK1_NNUNET_RESULTS_STAMP_NAME="${ROUND_STAMP}" \
      bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
    export TASK1_PREFLIGHT_STEP="iclr2026-baseline2-r${r}-train"
    bash "${ROOT}/scripts/task1_gpu_train_preflight.sh" || true
    sleep 2
    TASK1_CRASH_MONITOR_STAGE="baseline2_r${r}_train_running" \
    TASK1_NNUNET_RESULTS_STAMP_NAME="${ROUND_STAMP}" \
    TASK1_CRASH_MONITOR_ARM_SEC="${TASK1_CRASH_MONITOR_ARM_SEC:-86400}" \
      bash "${ROOT}/scripts/task1_crash_monitor_arm.sh" || true
    export TASK1_DOCKER_BACKGROUND=0
    bash "${ROOT}/other/task1_train_nnunet_from_dataset1.sh"
    TASK1_CRASH_MONITOR_STAGE="baseline2_r${r}_train_done" \
    TASK1_NNUNET_RESULTS_STAMP_NAME="${ROUND_STAMP}" \
      bash "${ROOT}/scripts/task1_crash_monitor_disarm.sh" || true
  fi

  # 若仍缺 best：可能被环境残留 BACKGROUND=1 提前返回，轮询等训完
  if [[ ! -f "${ROUND_DIR}/checkpoint_best.pth" ]]; then
    echo "[baseline2] waiting for checkpoint_best under ${ROUND_DIR} ..." >&2
    _cidf="${VIS}/docker_train_latest.cid"
    for _i in $(seq 1 720); do
      if [[ -f "${ROUND_DIR}/checkpoint_final.pth" && -f "${ROUND_DIR}/checkpoint_best.pth" ]]; then
        break
      fi
      if [[ -f "${_cidf}" ]]; then
        _cid="$(tr -d '[:space:]' <"${_cidf}" || true)"
        if [[ -n "${_cid}" ]] && docker inspect "${_cid}" >/dev/null 2>&1; then
          _st="$(docker inspect -f '{{.State.Status}}' "${_cid}" 2>/dev/null || true)"
          if [[ "${_st}" == "exited" || "${_st}" == "dead" ]]; then
            break
          fi
        fi
      fi
      sleep 10
    done
  fi

  [[ -f "${ROUND_DIR}/checkpoint_best.pth" ]] || {
    echo "[error] round ${r} missing checkpoint_best: ${ROUND_DIR}" >&2
    exit 1
  }
  CUR_BEST="${ROUND_DIR}/checkpoint_best.pth"
  CUR_RESULTS="${WORK}/nnUNet_results/${ROUND_STAMP}"
  # 在父目录保留当前 best 快捷方式
  mkdir -p "${UDA_ROOT}"
  ln -sfn "${CUR_BEST}" "${UDA_ROOT}/checkpoint_best_current.pth"
  _write_state "${r}" "round_done" "${CUR_BEST}" "${ROUND_STAMP}"

  {
    echo "round=${r}"
    echo "round_stamp=${ROUND_STAMP}"
    echo "best=${CUR_BEST}"
    echo "pseudo_labels=${LABELS_DIR}"
    echo "pseudo_b2nd=${SEG_B2ND}"
    echo "prior=${PRIOR_JSON}"
  } | tee -a "${ICLR_VIS}/iclr2026_baseline2_${PARENT_STAMP}_rounds.txt"
done

_write_state "$((ROUNDS - 1))" "done" "${CUR_BEST}" "${PARENT_STAMP}"
MANIFEST="${ICLR_VIS}/iclr2026_baseline2_${PARENT_STAMP}.txt"
{
  echo "job=iclr2026_baseline2_psma_uda_selftrain"
  echo "PARENT_STAMP=${PARENT_STAMP}"
  echo "FDG_BEST=${FDG_BEST}"
  echo "final_best=${CUR_BEST}"
  echo "rounds=${ROUNDS}"
  echo "epochs_per_round=${EP_PER_ROUND}"
  echo "train_iters=${TRAIN_ITERS} val_iters=${VAL_ITERS}"
  echo "lambda=${LAMBDA_START}->${LAMBDA_END}"
  echo "splits=${SPLITS_JSON}"
  echo "uda_root=${UDA_ROOT}"
  echo "state=${STATE_JSON}"
} | tee "${MANIFEST}"

echo "[baseline2] finished all ${ROUNDS} rounds"
echo "  final_best=${CUR_BEST}"
echo "  loss_png=${PARENT_LOSS_PNG}"
echo "  stop later: TASK1_NNUNET_RESULTS_STAMP_NAME=${PARENT_STAMP} bash scripts/task1_stop_train_and_resume.sh"
docker stop "uda_combined_plotter_${PARENT_STAMP}" 2>/dev/null || true
