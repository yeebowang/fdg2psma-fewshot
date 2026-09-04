#!/usr/bin/env bash
# Download ONLY the LesionTracer *pretrain* init (Dataset619 MultiTalent).
# FORBIDDEN for board: Zenodo 14007247 final LesionTracer, EDT, LocalEdit submission ckpts.
#
#   bash ICLR2026/run/download_dataset619_multitalent_pretrain_bg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${TASK1_DATASET619_DIR:-${ROOT}/ICLR2026/weights/Dataset619_nativemultistem}"
URL="${TASK1_DATASET619_URL:-https://zenodo.org/api/records/13753413/files/MultiTalentV2_challengeversion.zip/content}"
LOG="${ROOT}/ICLR2026/vis/nohup_download_dataset619_multitalent.log"
mkdir -p "${OUT}" "${ROOT}/ICLR2026/vis"
exec > >(tee -a "${LOG}") 2>&1

echo "[dataset619] $(date '+%F %T') → ${OUT}"
ZIP="${OUT}/MultiTalentV2_challengeversion.zip"
# Zenodo record size (bytes); used to detect incomplete downloads
EXPECT_BYTES="${TASK1_DATASET619_EXPECT_BYTES:-2325197563}"
rm -f "${ZIP}.partial" "${ZIP}.part.partial" 2>/dev/null || true
_need_dl=1
if [[ -f "${ZIP}" ]]; then
  _have="$(stat -c%s "${ZIP}" 2>/dev/null || echo 0)"
  if [[ "${_have}" -eq "${EXPECT_BYTES}" ]]; then
    echo "[dataset619] zip OK size=${_have} ($(du -h "${ZIP}" | awk '{print $1}'))"
    _need_dl=0
  else
    echo "[dataset619] incomplete zip ${_have}/${EXPECT_BYTES} → re-download with curl -C -"
    rm -f "${ZIP}"
  fi
fi
if [[ "${_need_dl}" == "1" ]]; then
  echo "[dataset619] curl resume → ${ZIP}"
  # -L follow redirects; -C - resume; --retry for transient errors
  curl -L --fail --retry 20 --retry-delay 5 --retry-all-errors \
    -C - --progress-bar \
    -o "${ZIP}" "${URL}"
  _have="$(stat -c%s "${ZIP}" 2>/dev/null || echo 0)"
  [[ "${_have}" -eq "${EXPECT_BYTES}" ]] || {
    echo "[error] size mismatch after curl: ${_have} != ${EXPECT_BYTES}" >&2
    exit 1
  }
  echo "[dataset619] downloaded $(du -h "${ZIP}" | awk '{print $1}')"
fi
# extract if checkpoint missing
CKPT="$(find "${OUT}" -name 'checkpoint_final.pth' 2>/dev/null | head -1 || true)"
if [[ -z "${CKPT}" ]]; then
  echo "[dataset619] extracting…"
  python3 - <<PY
import zipfile
from pathlib import Path
z=Path("${ZIP}")
out=Path("${OUT}")
with zipfile.ZipFile(z) as zf:
    zf.extractall(out)
print("extracted to", out)
PY
fi
CKPT="$(find "${OUT}" -name 'checkpoint_final.pth' | head -1)"
[[ -n "${CKPT}" && -f "${CKPT}" ]] || { echo "[error] checkpoint_final.pth not found under ${OUT}" >&2; exit 1; }
# refuse accidental final LesionTracer tree
if echo "${CKPT}" | grep -qiE '14007247|LesionTracer|autoPET-3-LesionTracer'; then
  echo "[error] refused path looks like FINAL LesionTracer: ${CKPT}" >&2
  exit 2
fi
echo "${CKPT}" > "${OUT}/PRETRAIN_CHECKPOINT.txt"
echo "[dataset619] OK pretrain_ckpt=${CKPT}"
# hard ban list for board pipelines
cat > "${OUT}/BOARD_FORBIDDEN_WEIGHTS.txt" <<'EOF'
# These MUST NOT be used for ICLR2026 aligned board scoring / training init:
Zenodo 14007247 autoPET-3-LesionTracer.zip (final submission K0)
EDT checkpoint_final.pth (BIRTH interactive)
LocalEdit FDG/PSMA tgz from weights-v0.3.0 (YixinChen interactive)
Any GC final Docker baked champion ensemble used as board metric source
# Allowed pretrain init only:
Zenodo 13753413 Dataset619_nativemultistem / MultiTalentV2_challengeversion.zip
EOF
echo "[dataset619] DONE $(date '+%F %T')"
