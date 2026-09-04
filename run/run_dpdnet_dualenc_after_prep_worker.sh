#!/usr/bin/env bash
# After CPU 250/251 preprocess DONE: adapt plans → CT+PET encoder → dual-enc FDG.
# Does not start training until TASK1_DPDNET_ENC_PREP_DONE.txt exists.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
ENC_ROOT="${CTRL}/ICLR2026/3D-MAE-PET-CT/weights/dpdnet"
IMAGE="${TASK1_NNUNET_IMAGE:-autopet_baseline:latest}"
DONE="${VIS}/TASK1_DPDNET_ENC_PREP_DONE.txt"
PP250="${WORK}/nnUNet_preprocessed/Dataset250_DpDNet_FDG_CT1ch"
PP251="${WORK}/nnUNet_preprocessed/Dataset251_DpDNet_FDG_PET1ch"

echo "[after-prep] wait DONE $(date '+%F %T')"
while [[ ! -f "${DONE}" ]]; do
  if ! pgrep -f "run_dpdnet_dualenc_encoder_cpu_prep_worker.sh" >/dev/null; then
    n250="$(find "${PP250}" -name '*.b2nd' 2>/dev/null | wc -l)"
    n251="$(find "${PP251}" -name '*.b2nd' 2>/dev/null | wc -l)"
    if [[ -f "${PP250}/nnUNetPlans.json" && -f "${PP251}/nnUNetPlans.json" && "${n250}" -ge 1600 && "${n251}" -ge 1600 ]]; then
      echo "[after-prep] prep worker gone but data complete (250=${n250} 251=${n251}) — continue"
      echo "status=ok recovered=$(date -Iseconds)" > "${DONE}"
      break
    fi
    echo "[after-prep] prep worker dead and data incomplete (250=${n250} 251=${n251}) — abort" >&2
    exit 1
  fi
  sleep 30
done
echo "[after-prep] prep DONE $(date '+%F %T')"

python3 "${CTRL}/ICLR2026/scripts/adapt_nnunet_plans_for_dpdnet.py" \
  "${PP250}/nnUNetPlans.json" \
  "${PP251}/nnUNetPlans.json"

python3 - "${PP250}/nnUNetPlans.json" "${PP251}/nnUNetPlans.json" <<'PY'
import json, sys
from pathlib import Path
ok = True
for p in sys.argv[1:]:
    cfg = json.loads(Path(p).read_text())["configurations"]["3d_fullres"]
    if "pool_op_kernel_sizes" not in cfg or "conv_kernel_sizes" not in cfg:
        print(f"[after-prep] plans still missing legacy keys: {p}", file=sys.stderr)
        ok = False
if not ok:
    raise SystemExit(2)
print("[after-prep] plans compatible")
PY

_enc_width_ok() {
  local ckpt="$1"
  docker run --rm --entrypoint python3 \
    -v "${CTRL}:${CTRL}" "${IMAGE}" \
    "${CTRL}/ICLR2026/scripts/check_dpdnet_encoder_width.py" \
    --ckpt "${ckpt}" --expect-out-ch 16
}

ENC_CT="${ENC_ROOT}/best_encoder_ct_epoch_94.pth"
ENC_PET="${ENC_ROOT}/best_encoder_pet_epoch_94.pth"
SKIP_RETRAIN=0
if [[ -f "${ENC_CT}" && -f "${ENC_PET}" ]]; then
  if _enc_width_ok "${ENC_CT}" && _enc_width_ok "${ENC_PET}"; then
    echo "[after-prep] encoder weights present and 16-ch — skip retrain"
    SKIP_RETRAIN=1
  else
    echo "[after-prep] encoder weights exist but width != 16 — retrain STUNetTrainer_small_prompt"
  fi
fi
if [[ "${SKIP_RETRAIN}" -eq 0 ]]; then
  echo "[after-prep] start CT encoder GPU0 + PET encoder GPU1"
  python3 "${CTRL}/ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py" \
    --board "${VIS}/iclr2026_aligned_fdg_fs50_f258_board.json" --no-plot \
    --patch-json '{"methods":{"dpdnet_dualenc":{"fdg_pretrain":{"status":"pending","note":"2GPU encoder retrain CT0+PET1"}}},"updated_note":"dual-enc encoder retrain after CPU prep"}' || true
  MODALITY=ct GPU_ID=0 bash "${CTRL}/ICLR2026/run/run_dpdnet_dualenc_retrain_encoders.sh" &
  p_ct=$!
  MODALITY=pet GPU_ID=1 bash "${CTRL}/ICLR2026/run/run_dpdnet_dualenc_retrain_encoders.sh" &
  p_pet=$!
  wait "${p_ct}" || echo "[after-prep] CT encoder rc=$?"
  wait "${p_pet}" || echo "[after-prep] PET encoder rc=$?"
fi

if [[ ! -f "${ENC_ROOT}/best_encoder_ct_epoch_94.pth" || ! -f "${ENC_ROOT}/best_encoder_pet_epoch_94.pth" ]]; then
  echo "[error] encoder weights missing after retrain" >&2
  ls -la "${ENC_ROOT}" >&2 || true
  exit 1
fi
if ! _enc_width_ok "${ENC_ROOT}/best_encoder_ct_epoch_94.pth" || ! _enc_width_ok "${ENC_ROOT}/best_encoder_pet_epoch_94.pth"; then
  echo "[error] encoder weights still not 16-ch after retrain" >&2
  exit 1
fi

echo "[after-prep] encoders ready → dual-enc FDG→PSMA"
TASK1_DPDNET_GPU=0 TASK1_BOARD_METHOD=dpdnet_dualenc \
  bash "${CTRL}/ICLR2026/run/run_dpdnet_dualenc_aligned_fdg_psma_bg.sh"
echo "[after-prep] ALL DONE $(date '+%F %T')"
