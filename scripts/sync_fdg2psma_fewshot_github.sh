#!/usr/bin/env bash
# Curate + sync aligned board code to https://github.com/yeebowang/fdg2psma-fewshot
# - Syncs only board-protocol essentials
# - Deletes cascade/competition/one-off cruft from the public tree
# - Refreshes README assets from the live board JSON
# - Defaults: all RNG seeds = 42 (see run/_aligned_seed_env.sh)
#
#   bash ICLR2026/scripts/sync_fdg2psma_fewshot_github.sh [--no-push]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${ROOT}/ICLR2026"
DST="${FDG2PSMA_DST:-${ROOT}/_publish_fdg2psma_fewshot}"
BOARD_JSON="${TASK1_ALIGN_BOARD_JSON:-${SRC}/vis/iclr2026_aligned_fdg_fs50_f258_board.json}"
NO_PUSH=0
[[ "${1:-}" == "--no-push" ]] && NO_PUSH=1

if [[ ! -d "${DST}/.git" ]]; then
  rm -rf "${DST}"
  git clone https://github.com/yeebowang/fdg2psma-fewshot.git "${DST}"
else
  git -C "${DST}" fetch origin
  git -C "${DST}" checkout main
  git -C "${DST}" reset --hard origin/main
fi

echo "[sync] DST=${DST}"

rsync -a --delete \
  --exclude '__pycache__/' --exclude '*.pyc' \
  "${SRC}/scripts/" "${DST}/scripts/"

rsync -a --delete \
  --exclude '__pycache__/' \
  "${SRC}/run/" "${DST}/run/"

rsync -a --delete \
  --exclude 'proto_retrieval/*.npz' \
  --exclude 'proto_retrieval/*.pkl' \
  "${SRC}/data/" "${DST}/data/"

rsync -a --delete \
  --exclude 'runs/' --exclude 'weights/' --exclude '__pycache__/' \
  --exclude '*.pth' --exclude '*.pt' --exclude '*.ckpt' \
  "${SRC}/3D-MAE-PET-CT/" "${DST}/3D-MAE-PET-CT/"

mkdir -p "${DST}/nnunet_ext_trainers"
rsync -a \
  "${ROOT}/nnunet_ext_trainers/nnUNetTrainer_Task1StdTrainVal50.py" \
  "${ROOT}/nnunet_ext_trainers/task1_ram_shard.py" \
  "${ROOT}/nnunet_ext_trainers/task1_pseudo_seg_dataset.py" \
  "${ROOT}/nnunet_ext_trainers/task1_intensity_da_protect.py" \
  "${ROOT}/nnunet_ext_trainers/task1_weighted_mae_loss.py" \
  "${DST}/nnunet_ext_trainers/"

# Drop non-aligned trainers if previously published
rm -f "${DST}/nnunet_ext_trainers/"nnUNetTrainer_Task1{Dataset2PairedFinetune,DemoSteps,MambaKD,PETVSchedule,StepBatch}.py
rm -f "${DST}/nnunet_ext_trainers/"nnUNetTrainer_Task1StdTrainVal50_{3chCascadeRef,5chCascadeClickProtect,5chCascadeProb,5chCascadeWMAE,ClickProtect}.py
rm -f "${DST}/nnunet_ext_trainers/"task1_{3ch_protect_seg_da,3ch_split_ref_dataset,5ch_split_prob_dataset}.py

# Third-party: SegAnyPET + DpDNet only
mkdir -p "${DST}/third_party"
rsync -a --delete \
  --exclude '__pycache__/' --exclude '*.pth' --exclude '*.pt' \
  --exclude 'runs/' --exclude 'checkpoints/' --exclude '.git/' \
  "${SRC}/third_party/SegAnyPET/" "${DST}/third_party/SegAnyPET/"
rsync -a --delete \
  --exclude '__pycache__/' --exclude '*.pth' --exclude '*.pt' \
  --exclude 'runs/' --exclude 'checkpoints/' --exclude '.git/' \
  "${SRC}/third_party/DpDNet/" "${DST}/third_party/DpDNet/"
rm -rf "${DST}/third_party/autoPET-V-BIRTH-final" \
  "${DST}/third_party/autopet-v-yixinchen" \
  "${DST}/third_party/seganypet_pip" \
  "${DST}/third_party/README_competition_repos.md"

cat > "${DST}/third_party/README.md" <<'EOF'
# Third-party method mirrors

Local checkouts used by the aligned FDG→PSMA board:

- `SegAnyPET/` — SegAnyPET (ICCV'25)
- `DpDNet/` — DpDNet (MICCAI'25)

Upstream licenses apply. Competition-challenge forks are **not** shipped here.
EOF

# Paper (anonymous draft)
rsync -a --delete \
  --exclude '__pycache__/' --exclude '*.aux' --exclude '*.log' --exclude '*.out' \
  --exclude '*.bbl' --exclude '*.blg' --exclude '*.synctex.gz' \
  "${SRC}/paper/" "${DST}/paper/" 2>/dev/null || true
if [[ -d "${SRC}/docs" ]]; then
  rsync -a --delete --exclude '*.pdf' --exclude '*.docx' \
    "${SRC}/docs/" "${DST}/docs/" 2>/dev/null || true
fi

# ---- Delete redundant run one-offs (local machine / resume chains) ----
cd "${DST}/run"
rm -f \
  chain_seganypet_bs3_then_baseline1_eval_f258_bg.sh \
  chain_seganypet_then_official_bg.sh \
  download_dataset619_multitalent_pretrain_bg.sh \
  _launch_mae_fullcase_once.sh \
  run_aligned_board_queue_keeper_bg.sh \
  run_aligned_continue_after_mae_nnunet_first_bg.sh \
  run_aligned_pipeline_watchdog_bg.sh \
  run_aligned_resume_after_mae_test20_bg.sh \
  run_aligned_resume_after_nnunet_fdg_done_bg.sh \
  run_dpdnet_dualenc_continue_after_fs50_bg.sh \
  run_dpdnet_fdg_bs_ladder_bg.sh \
  run_f258_resume_100ep_until_val_decline_bg.sh \
  run_mae_psma_ssl_continued_then_fewshot10_bg.sh \
  run_mae_psma_ssl_gentle_fewshot50_f258_bg.sh \
  run_remaining_gpu_idle_watchdog.sh \
  run_chenyixin_pretrained_fs50_seed42_bg.sh \
  run_chenyixin_pretrained_fc70_seed42_bg.sh \
  run_competition_*.sh \
  run_*competition*.sh \
  run_gpu013_auto_*.sh \
  run_hemingduo_*.sh \
  run_chenyixin_*.sh \
  2>/dev/null || true

# ---- Delete redundant progress-plot / UDA-only helpers from public scripts ----
cd "${DST}/scripts"
rm -f \
  assert_competition_board_weights.py \
  backfill_competition_fp_fn_from_score_detail.py \
  baseline1_eval_9fold_progress_plot.py \
  mae_fewshot50_9fold_progress_plot.py \
  mae_gentle_f258_progress_plot.py \
  mae_monai_f258_progress_plot.py \
  nnunet_psma_fs50_f258_progress_plot.py \
  seganypet_f258_progress_plot.py \
  psma_uda_combined_loss_plot.py \
  psma_uda_live_status_plot.py \
  psma_uda_make_pseudo_labels.py \
  psma_uda_predict_train.sh \
  psma_uda_pseudo_nii_to_seg_b2nd.py \
  2>/dev/null || true

# Seed policy note
cat > "${DST}/SEED.md" <<'EOF'
# Reproducibility seeds

Aligned board protocol uses **seed = 42** everywhere by default:

| Knob | Default | Where |
|------|--------:|-------|
| Stratified / few-shot split export | 42 | `scripts/make_stratified_splits_dataset1.py`, `export_mae_psma_fewshot*.py` |
| `TASK1_TRAIN_SEED` | 42 | `nnUNetTrainer_Task1StdTrainVal50` (python/numpy/torch/CUDA) |
| `TASK1_RAM_SHARD_SEED` | 42 | `task1_ram_shard.py` |
| `TASK1_FEWSHOT_SPLIT_SEED` | 42 | `run/_aligned_seed_env.sh` |

Override only when intentionally ablating seed; set `TASK1_TRAIN_SEED=none` to disable the trainer global seed.
EOF

# PROTOCOL: keep local notes but add seed banner; strip nothing critical
if [[ -f "${SRC}/PROTOCOL.md" ]]; then
  cp -a "${SRC}/PROTOCOL.md" "${DST}/PROTOCOL.md"
fi

# Refresh sanitized README table from live board
mkdir -p "${DST}/assets"
EXPORT=(python3 "${SRC}/scripts/export_readme_board.py"
  --board "${BOARD_JSON}"
  --out-json "${DST}/assets/board_summary.json"
  --out-md "${DST}/assets/board_table.md")
if command -v docker >/dev/null 2>&1 && docker image inspect iclr2026_3dmae_petct:cu118 >/dev/null 2>&1; then
  docker run --rm \
    -v "${ROOT}:${ROOT}" -v "${DST}:${DST}" \
    -u "$(id -u):$(id -g)" \
    iclr2026_3dmae_petct:cu118 \
    python3 "${SRC}/scripts/export_readme_board.py" \
      --board "${BOARD_JSON}" \
      --out-json "${DST}/assets/board_summary.json" \
      --out-md "${DST}/assets/board_table.md" \
      --png "${DST}/assets/progress_board.png" || \
  python3 "${SRC}/scripts/export_readme_board.py" \
    --board "${BOARD_JSON}" \
    --out-json "${DST}/assets/board_summary.json" \
    --out-md "${DST}/assets/board_table.md"
else
  python3 "${SRC}/scripts/export_readme_board.py" \
    --board "${BOARD_JSON}" \
    --out-json "${DST}/assets/board_summary.json" \
    --out-md "${DST}/assets/board_table.md"
fi

# README
TABLE_MD="$(cat "${DST}/assets/board_table.md")"
cat > "${DST}/README.md" <<EOF
# fdg2psma-fewshot

Code for an **ICLR workshop** study on **FDG → PSMA** PET/CT lesion segmentation under **few-shot / zero-shot** transfer, comparing supervised baselines with representation-learning initializations (MAE, MIM, dual-encoder, etc.).

Aligned protocol: shared FDG supervised stage → PSMA \`fs50\` / \`fs10\` / \`fs5\` / \`fs0\` (+ \`fc70%\` and FDG TEST).

**Reproducibility:** all random seeds default to **42** (see [SEED.md](SEED.md)).

## Results (mean TEST metrics)

Each metric cell is three lines — **Dice / FP / FN (%)** — also shown in the column headers. Dice excludes empty-GT cases; FP = FP/Neg, FN = FN/Pos (voxel micro-average). Numeric snapshot: [\`assets/board_summary.json\`](assets/board_summary.json).

${TABLE_MD}

Refresh the table from a live board JSON:

\`\`\`bash
python3 scripts/export_readme_board.py \\
  --board /path/to/iclr2026_aligned_fdg_fs50_f258_board.json \\
  --out-json assets/board_summary.json \\
  --out-md assets/board_table.md
\`\`\`

## Workshop paper draft

ICLR-style LaTeX draft (anonymous): [\`paper/\`](paper/) (\`main.tex\`, mean-metric figure, references).

\`\`\`bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
\`\`\`

## Repository layout

| Path | Contents |
|------|----------|
| \`scripts/\` | Aligned board, split export, eval / ingest helpers |
| \`run/\` | Background launchers for FDG / PSMA few-shot stages (\`_aligned_seed_env.sh\`) |
| \`data/\` | Stratified & few-shot split JSONs (no NIfTI) |
| \`3D-MAE-PET-CT/\` | PET/CT MAE + downstream code (**no** \`runs/\` / weight blobs) |
| \`third_party/\` | SegAnyPET + DpDNet mirrors |
| \`nnunet_ext_trainers/\` | \`Task1StdTrainVal50\` + RAM-shard helpers |
| \`PROTOCOL.md\` | Experiment protocol notes |
| \`SEED.md\` | Seed=42 policy |
| \`assets/\` | Sanitized mean-metric JSON + markdown table |

**Not** published: raw PET/CT volumes, training caches, checkpoints, per-fold prediction shards, cascade/competition trainers, or local resume/chain scripts.

## Data & compute notes

- Imaging data follow the AutoPET-style PET/CT layout used in our internal \`dataset1\` stratified split (\`data/splits_stratified_70_10_20.json\`).
- Many launchers assume env vars such as \`TASK1_BASE\` / a workspace root; adapt absolute paths in \`scripts/iclr2026_aligned_fdg_fs50_board.py\` before running locally.
- Third-party code retains upstream licenses; see each subdirectory.

## License

Apache-2.0 (see [LICENSE](LICENSE)), except where \`third_party/\` components specify otherwise.
EOF

# Ensure LICENSE / gitignore
[[ -f "${DST}/LICENSE" ]] || cp -a "${ROOT}/LICENSE" "${DST}/LICENSE" 2>/dev/null || true
if [[ ! -f "${DST}/.gitignore" ]]; then
  cat > "${DST}/.gitignore" <<'EOF'
__pycache__/
*.py[cod]
*.pth
*.pt
*.ckpt
**/runs/
**/nnUNet_results/
*.nii
*.nii.gz
.env
third_party/seganypet_pip/
EOF
fi

cd "${DST}"
# drop empty dirs
find . -type d -empty ! -path './.git/*' -delete 2>/dev/null || true

echo "[sync] file count=$(find . -type f ! -path './.git/*' | wc -l) size=$(du -sh . | awk '{print $1}')"
git add -A
git status -sb | head -60

if git diff --cached --quiet; then
  echo "[sync] no changes to commit"
  exit 0
fi

git commit -m "$(cat <<'EOF'
Curate aligned board code: seed=42 defaults, drop redundant trainers/scripts.

Sync StdTrainVal50 + board/run/data essentials; refresh README metrics; remove cascade/competition/one-off resume helpers from the public tree.
EOF
)"

if [[ "${NO_PUSH}" == "1" ]]; then
  echo "[sync] skipped push (--no-push)"
  exit 0
fi

git push -u origin HEAD
echo "[sync] pushed → https://github.com/yeebowang/fdg2psma-fewshot"
