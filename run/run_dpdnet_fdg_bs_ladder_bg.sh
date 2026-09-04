#!/usr/bin/env bash
# DpDNet FDG · OOM ladder (nnUNetv2 DDP: plans batch_size = GLOBAL):
#   1) 1 free GPU among 0/1/3 · global_bs=6 (=6/GPU)
#   2) OOM → 2 free GPUs · global_bs=6 (=3/GPU)
#   3) OOM → 3 GPUs 0,1,3 · global_bs=6 (=2/GPU)
#
# Requires Dataset239 npz ready. Clears STOP for the new STAMP only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${WORK}/01_train_vis"
ICLR_VIS="${CTRL}/ICLR2026/vis"
DPD="${CTRL}/ICLR2026/third_party/DpDNet"
IMAGE="${TASK1_NNUNET_IMAGE:-autopet_baseline:latest}"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${ICLR_VIS}/iclr2026_aligned_fdg_fs50_f258_board.json}"

DATASET_ID="${DATASET_ID:-239}"
DS="Dataset${DATASET_ID}_DpDNet_FDG_2ch"
TRAINER="${TRAINER:-STUNetTrainer_small_prompt}"
CONFIG="${CONFIG:-3d_fullres}"
FOLD="${FOLD:-0}"
GLOBAL_BS="${TASK1_DPDNET_BATCH_SIZE:-6}"
TOTAL_EPOCHS="${TASK1_DPDNET_NUM_EPOCHS:-169}"
TRAIN_ITERS="${TASK1_DPDNET_TRAIN_ITERS:-${TASK1_TRAIN_ITERS_PER_EPOCH:-70}}"
VAL_ITERS="${TASK1_DPDNET_VAL_ITERS:-${TASK1_VAL_ITERS_PER_EPOCH:-0}}"
N_PROC_DA="${TASK1_DPDNET_N_PROC_DA:-4}"
POOL_GPUS_CSV="${TASK1_DPDNET_GPU_POOL:-0,1,3}"
CFG_DIR="${WORK}/nnUNet_preprocessed/${DS}/nnUNetPlans_3d_fullres"
PLANS="${WORK}/nnUNet_preprocessed/${DS}/nnUNetPlans.json"
POLL_BOOT="${TASK1_DPDNET_BOOT_POLL_SEC:-90}"

mkdir -p "${VIS}" "${ICLR_VIS}"

_npz_ready() {
  local n
  n="$(ls "${CFG_DIR}"/*.npz 2>/dev/null | wc -l)"
  [[ "${n}" -ge 800 ]]
}

_free_gpus() {
  local pool g busy ids name
  IFS=',' read -r -a pool <<< "${POOL_GPUS_CSV}"
  for g in "${pool[@]}"; do
    busy=0
    while read -r id; do
      [[ -n "${id}" ]] || continue
      ids="$(docker inspect -f '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{.}} {{end}}{{end}}' "${id}" 2>/dev/null || true)"
      for x in $ids; do
        if [[ "$x" == "$g" ]]; then
          busy=1
          break
        fi
      done
      [[ "${busy}" -eq 0 ]] || break
    done < <(docker ps -q)
    if [[ "${busy}" -eq 0 ]]; then
      echo -n "${g} "
    fi
  done
}

_set_plans_bs() {
  python3 - <<PY
import json
from pathlib import Path
p = Path("${PLANS}")
d = json.loads(p.read_text())
d.setdefault("configurations", {}).setdefault("3d_fullres", {})["batch_size"] = int("${GLOBAL_BS}")
p.write_text(json.dumps(d, indent=2) + "\n")
print("[dpdnet-ladder] plans batch_size=${GLOBAL_BS}")
PY
}

_is_oom() {
  local log="$1"
  [[ -f "${log}" ]] && grep -qiE 'OutOfMemoryError|CUDA out of memory|torch.cuda.OutOfMemoryError' "${log}"
}

_wait_npz() {
  echo "[dpdnet-ladder] wait npz under ${CFG_DIR}"
  while ! _npz_ready; do
    echo "[dpdnet-ladder] npz=$(ls "${CFG_DIR}"/*.npz 2>/dev/null | wc -l) $(TZ=Asia/Shanghai date +%H:%M:%S)"
    sleep 60
  done
  echo "[dpdnet-ladder] npz ready $(ls "${CFG_DIR}"/*.npz 2>/dev/null | wc -l)"
}

_run_attempt() {
  local n_gpu="$1"
  shift
  local gpus=("$@")
  local gpu_csv
  gpu_csv="$(IFS=,; echo "${gpus[*]}")"
  local stamp tag
  stamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_iclr2026_dpdnet_fdg_${n_gpu}gpu_gbs${GLOBAL_BS}_${TOTAL_EPOCHS}ep_gpu${gpu_csv//,/}"
  tag="${n_gpu}gpu gbs=${GLOBAL_BS} (≈$((GLOBAL_BS / n_gpu))/GPU) devices=${gpu_csv}"

  export TASK1_NNUNET_RESULTS_STAMP_NAME="${stamp}"
  export TASK1_BASE="${DATA}"
  local results="${WORK}/nnUNet_results/${stamp}"
  local log="${ICLR_VIS}/nohup_dpdnet_fdg_${stamp}.log"
  local cname="dpdnet_fdg_${stamp}"
  mkdir -p "${results}"
  # container runs as uid=algorithm(999); host-created dirs must be world-writable
  chmod -R a+rwX "${results}" || true
  docker run --rm --user root -v "${DATA}:${DATA}" --entrypoint bash "${IMAGE}" -lc \
    "mkdir -p '${results}' && chmod -R a+rwX '${results}'" >/dev/null 2>&1 || true
  rm -f "${VIS}/TASK1_TRAIN_STOP_${stamp}.txt"

  local inner_cvd
  if [[ "${n_gpu}" -eq 1 ]]; then
    inner_cvd=0
  else
    inner_cvd="$(seq -s, 0 $((n_gpu - 1)))"
  fi

  export TASK1_PREFLIGHT_GPUS="${gpus[*]}"
  export TASK1_PREFLIGHT_LABEL="iclr2026-dpdnet-fdg-ladder"
  bash "${CTRL}/scripts/task1_gpu_train_preflight.sh" || true
  bash "${CTRL}/scripts/task1_crash_monitor_disarm.sh" || true

  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${BOARD_JSON}" \
    --patch-json "{\"methods\":{\"dpdnet\":{\"fdg_pretrain\":{\"status\":\"running\",\"stamp\":\"${stamp}\",\"bs\":${GLOBAL_BS},\"bs_note\":\"${tag}\",\"total_epochs\":${TOTAL_EPOCHS}}}},\"updated_note\":\"DpDNet FDG try ${tag}\"}" || true

  docker rm -f "${cname}" >/dev/null 2>&1 || true
  echo "[dpdnet-ladder] TRY ${tag} STAMP=${stamp}"
  nohup docker run --rm \
    --name "${cname}" \
    --gpus "\"device=${gpu_csv}\"" \
    -e CUDA_VISIBLE_DEVICES="${inner_cvd}" \
    -e HOME=/home/algorithm \
    -e TASK1_DPDNET_NUM_EPOCHS="${TOTAL_EPOCHS}" \
    -e nnUNet_raw="${WORK}/nnUNet_raw" \
    -e nnUNet_preprocessed="${WORK}/nnUNet_preprocessed" \
    -e nnUNet_results="${results}" \
    -e PYTHONPATH="${DPD}:/home/algorithm/.local/lib/python3.11/site-packages" \
    -e nnUNet_n_proc_DA="${N_PROC_DA}" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "${CTRL}:${CTRL}" \
    -v "${DATA}:${DATA}" \
    --shm-size=16g \
    --entrypoint bash \
    "${IMAGE}" \
    -lc "mkdir -p '${results}' && python3 -c 'import nnunetv2,nnunetv2.training.nnUNetTrainer.STUNetTrainer as T; print(\"nnunet\", nnunetv2.__file__); print(\"stunet\", T.__file__)' && nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} -tr ${TRAINER} -num_gpus ${n_gpu}" \
    >"${log}" 2>&1 &
  echo $! > "${results}/nohup.pid"
  sleep 8
  bash "${CTRL}/scripts/task1_crash_monitor_arm.sh" || true

  # early OOM detection window
  local t=0
  while [[ "${t}" -lt "${POLL_BOOT}" ]]; do
    if ! docker ps --format '{{.Names}}' | grep -qx "${cname}"; then
      break
    fi
    if _is_oom "${log}"; then
      echo "[dpdnet-ladder] OOM detected early → stop ${cname}"
      docker rm -f "${cname}" >/dev/null 2>&1 || true
      break
    fi
    # healthy if epoch log appeared
    if grep -qE 'Epoch [0-9]+/|train_loss|Epoch time' "${log}" 2>/dev/null; then
      echo "[dpdnet-ladder] training bootstrapped OK (${tag})"
      wait "$(cat "${results}/nohup.pid")"
      local rc=$?
      if [[ "${rc}" -eq 0 ]] && ! _is_oom "${log}"; then
        echo "OK ${stamp}"
        return 0
      fi
      if _is_oom "${log}"; then
        echo "OOM ${stamp}"
        return 2
      fi
      echo "FAIL ${stamp} rc=${rc}"
      return 1
    fi
    sleep 5
    t=$((t + 5))
  done

  if docker ps --format '{{.Names}}' | grep -qx "${cname}"; then
    # still running past boot window → treat as success path (wait finish)
    echo "[dpdnet-ladder] still running after ${POLL_BOOT}s → wait finish"
    wait "$(cat "${results}/nohup.pid")"
    local rc=$?
    if [[ "${rc}" -eq 0 ]] && ! _is_oom "${log}"; then
      echo "OK ${stamp}"
      return 0
    fi
    if _is_oom "${log}"; then
      echo "OOM ${stamp}"
      return 2
    fi
    echo "FAIL ${stamp} rc=${rc}"
    return 1
  fi

  # container exited during boot
  wait "$(cat "${results}/nohup.pid")" 2>/dev/null || true
  if _is_oom "${log}"; then
    echo "OOM ${stamp}"
    return 2
  fi
  echo "FAIL ${stamp} (exited early)"
  return 1
}

_wait_npz
_set_plans_bs

echo "[dpdnet-ladder] pool=${POOL_GPUS_CSV} global_bs=${GLOBAL_BS}"

FORCE_TIER="${TASK1_DPDNET_FORCE_TIER:-0}"
if [[ "${FORCE_TIER}" == "3" ]]; then
  # Exclusive 3-GPU queue: skip OOM ladder, always gbs=6 on 0,1,3 (=2/GPU)
  IFS=',' read -r -a FORCE_GPUS <<< "${TASK1_DPDNET_FORCE_GPUS:-${POOL_GPUS_CSV}}"
  echo "[dpdnet-ladder] FORCE_TIER=3 gpus=${FORCE_GPUS[*]}"
  while true; do
    read -r -a FREE <<< "$(_free_gpus)"
    echo "[dpdnet-ladder] free=[${FREE[*]}] need=${#FORCE_GPUS[@]} for force-tier3"
    ok=1
    for g in "${FORCE_GPUS[@]}"; do
      found=0
      for f in "${FREE[@]}"; do
        [[ "$f" == "$g" ]] && found=1 && break
      done
      [[ "${found}" -eq 1 ]] || ok=0
    done
    if [[ "${ok}" -eq 1 ]]; then
      set +e
      _run_attempt 3 "${FORCE_GPUS[@]}"
      rc=$?
      set -e
      if [[ "${rc}" -eq 0 ]]; then
        python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
          --board "${BOARD_JSON}" \
          --patch-json "{\"methods\":{\"dpdnet\":{\"fdg_pretrain\":{\"status\":\"done\",\"bs_note\":\"3gpu gbs=${GLOBAL_BS} (=2/GPU) force\"}}},\"updated_note\":\"DpDNet FDG done 3gpu force\"}" || true
        exit 0
      fi
      echo "[dpdnet-ladder] force-tier3 failed rc=${rc}" >&2
      exit "${rc}"
    fi
    sleep 45
  done
fi

# Tier 1: 1 GPU bs=6
while true; do
  read -r -a FREE <<< "$(_free_gpus)"
  echo "[dpdnet-ladder] free=[${FREE[*]}] need>=1 for tier1"
  if [[ "${#FREE[@]}" -ge 1 ]]; then
    set +e
    _run_attempt 1 "${FREE[0]}"
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
        --board "${BOARD_JSON}" \
        --patch-json "{\"methods\":{\"dpdnet\":{\"fdg_pretrain\":{\"status\":\"done\",\"bs_note\":\"1gpu gbs=${GLOBAL_BS}\"}}},\"updated_note\":\"DpDNet FDG done 1gpu\"}" || true
      exit 0
    fi
    if [[ "${rc}" -ne 2 ]]; then
      echo "[dpdnet-ladder] non-OOM fail at tier1" >&2
      exit "${rc}"
    fi
    echo "[dpdnet-ladder] tier1 OOM → escalate"
    break
  fi
  sleep 45
done

# Tier 2: 2 GPUs · 3/GPU
while true; do
  read -r -a FREE <<< "$(_free_gpus)"
  echo "[dpdnet-ladder] free=[${FREE[*]}] need>=2 for tier2"
  if [[ "${#FREE[@]}" -ge 2 ]]; then
    set +e
    _run_attempt 2 "${FREE[0]}" "${FREE[1]}"
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
        --board "${BOARD_JSON}" \
        --patch-json "{\"methods\":{\"dpdnet\":{\"fdg_pretrain\":{\"status\":\"done\",\"bs_note\":\"2gpu gbs=${GLOBAL_BS} (=3/GPU)\"}}},\"updated_note\":\"DpDNet FDG done 2gpu\"}" || true
      exit 0
    fi
    if [[ "${rc}" -ne 2 ]]; then
      echo "[dpdnet-ladder] non-OOM fail at tier2" >&2
      exit "${rc}"
    fi
    echo "[dpdnet-ladder] tier2 OOM → escalate"
    break
  fi
  sleep 45
done

# Tier 3: 3 GPUs · 2/GPU
while true; do
  read -r -a FREE <<< "$(_free_gpus)"
  echo "[dpdnet-ladder] free=[${FREE[*]}] need>=3 for tier3"
  if [[ "${#FREE[@]}" -ge 3 ]]; then
    set +e
    _run_attempt 3 "${FREE[0]}" "${FREE[1]}" "${FREE[2]}"
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
        --board "${BOARD_JSON}" \
        --patch-json "{\"methods\":{\"dpdnet\":{\"fdg_pretrain\":{\"status\":\"done\",\"bs_note\":\"3gpu gbs=${GLOBAL_BS} (=2/GPU)\"}}},\"updated_note\":\"DpDNet FDG done 3gpu\"}" || true
      exit 0
    fi
    echo "[dpdnet-ladder] tier3 failed rc=${rc}" >&2
    exit "${rc}"
  fi
  sleep 45
done
