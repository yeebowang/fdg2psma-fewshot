# AutoPET V competition codebases (board rows)

Local mirrors used by the ICLR2026 aligned progress board:

| Board keys | Upstream | Local path |
|---|---|---|
| `hemingduo_scratch`, `hemingduo` | https://github.com/hemingduo85-droid/autoPET-V-BIRTH-final | `autoPET-V-BIRTH-final/` |
| `chenyixin_scratch`, `chenyixin` | https://github.com/YixinChen-AI/autopet-v | `autopet-v-yixinchen/` |

## Board policy (hard)

These trees are Grand Challenge **interactive / submission** code. Aligned board cells must **not** be filled by scoring their final GC bake-in models.

| Row | Init | Cascade |
|---|---|---|
| `*_scratch` | random / scratch | **FDG** (tr70/val0 · 169ep) → PSMA fs50/10/5 → fs0 / fc70 / FDG TEST |
| `hemingduo` / `chenyixin` | **only** Zenodo [13753413](https://zenodo.org/records/13753413) `Dataset619_nativemultistem` (MultiTalent pretrain they load *before* FDG training) | same FDG → PSMA cascade |

### Forbidden on the progress board

- Zenodo **14007247** LesionTracer **final** checkpoint
- BIRTH **EDT** interactive finals
- YixinChen **LocalEdit / TACE** submission weights (`weights-v0.3.0`, HoleGuard Docker bake-in)
- Using GC zero-click / champion Docker K0 output as board Dice

### Allowed pretrain marker

```bash
bash ICLR2026/run/download_dataset619_multitalent_pretrain_bg.sh
# → ICLR2026/weights/Dataset619_nativemultistem/PRETRAIN_CHECKPOINT.txt
# → ICLR2026/weights/Dataset619_nativemultistem/BOARD_FORBIDDEN_WEIGHTS.txt
```

Policy gate:

```bash
python3 ICLR2026/scripts/assert_competition_board_weights.py --require-dataset619
TASK1_COMP_DRY_RUN=1 bash ICLR2026/run/run_competition_aligned_fdg_psma_queue_bg.sh
```

## Scratch queue (no Dataset619)

```bash
bash ICLR2026/run/run_competition_scratch_queue_bg.sh
# hemingduo_scratch → chenyixin_scratch；FDG169 → PSMA fs50 f258 → TEST20
```

Interim backbone: Dataset228 + `nnUNetTrainer_Task1StdTrainVal50`（对齐板协议）。MultiTalent 双头配方另接；**不用**最终提交权重。