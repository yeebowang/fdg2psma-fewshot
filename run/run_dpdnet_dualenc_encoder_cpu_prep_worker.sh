#!/usr/bin/env bash
# CPU worker: 1ch FDG raw + plan/preprocess for dual-enc retrain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CTRL="${ROOT}"
DATA="${TASK1_BASE:-/media/ybwang/data1/PSMA-DATA}"
WORK="${WORK_DIR:-${DATA}/task1_train_workspace}"
VIS="${CTRL}/ICLR2026/vis"
IMAGE="${TASK1_NNUNET_IMAGE:-autopet_baseline:latest}"
DONE="${VIS}/TASK1_DPDNET_ENC_PREP_DONE.txt"

echo "[enc-prep] start $(date '+%F %T')"
python3 "${CTRL}/ICLR2026/scripts/prepare_dpdnet_1ch_encoder_datasets.py"
# parent nnUNet_preprocessed is root-owned; create dest as root then world-writable
docker run --rm --user root -v "${DATA}:${DATA}" --entrypoint bash "${IMAGE}" -lc \
  "mkdir -p '${WORK}/nnUNet_preprocessed/Dataset250_DpDNet_FDG_CT1ch' \
            '${WORK}/nnUNet_preprocessed/Dataset251_DpDNet_FDG_PET1ch' \
   && chmod -R a+rwX \
        '${WORK}/nnUNet_preprocessed/Dataset250_DpDNet_FDG_CT1ch' \
        '${WORK}/nnUNet_preprocessed/Dataset251_DpDNet_FDG_PET1ch'"

docker run --rm \
  --user algorithm \
  -e HOME=/home/algorithm \
  -e nnUNet_raw="${WORK}/nnUNet_raw" \
  -e nnUNet_preprocessed="${WORK}/nnUNet_preprocessed" \
  -e nnUNet_results="${WORK}/nnUNet_results/_enc_prep" \
  -v "${CTRL}:${CTRL}" -v "${DATA}:${DATA}" \
  --entrypoint bash "${IMAGE}" \
  -lc 'set -euo pipefail; python3 -c "import nnunetv2; print(nnunetv2.__file__)"; nnUNetv2_plan_and_preprocess -d 250 -c 3d_fullres --verify_dataset_integrity; nnUNetv2_plan_and_preprocess -d 251 -c 3d_fullres --verify_dataset_integrity'

docker run --rm --user root -v "${DATA}:${DATA}" --entrypoint bash "${IMAGE}" -lc \
  "chown -R $(id -u):$(id -g) \
    '${WORK}/nnUNet_raw/Dataset250_DpDNet_FDG_CT1ch' \
    '${WORK}/nnUNet_raw/Dataset251_DpDNet_FDG_PET1ch' \
    '${WORK}/nnUNet_preprocessed/Dataset250_DpDNet_FDG_CT1ch' \
    '${WORK}/nnUNet_preprocessed/Dataset251_DpDNet_FDG_PET1ch' 2>/dev/null || true"

for ds in Dataset250_DpDNet_FDG_CT1ch Dataset251_DpDNet_FDG_PET1ch; do
  hint="${WORK}/nnUNet_raw/${ds}/splits_hint.json"
  dst="${WORK}/nnUNet_preprocessed/${ds}/splits_final.json"
  if [[ -f "${hint}" && -d "$(dirname "${dst}")" ]]; then
    cp -f "${hint}" "${dst}"
  fi
done

python3 "${CTRL}/ICLR2026/scripts/adapt_nnunet_plans_for_dpdnet.py" \
  "${WORK}/nnUNet_preprocessed/Dataset250_DpDNet_FDG_CT1ch/nnUNetPlans.json" \
  "${WORK}/nnUNet_preprocessed/Dataset251_DpDNet_FDG_PET1ch/nnUNetPlans.json"

echo "status=ok at=$(date -Iseconds)" > "${DONE}"
echo "[enc-prep] DONE $(date '+%F %T')"
