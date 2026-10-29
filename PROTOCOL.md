# Aligned FDG→PSMA board protocol

Experiment notes for the public board in this repository
([fdg2psma-fewshot](https://github.com/yeebowang/fdg2psma-fewshot)).
Splits and trainers default to **seed = 42** (see [SEED.md](SEED.md)).

## Data

- Imaging layout: AutoPET~III-style multicenter multitracer PET/CT
  (CT+PET, binary lesion masks; see Gatidis et al., *Sci. Data* 2022, and the
  autoPET~III challenge).
- Internal stratified split: `data/splits_stratified_70_10_20.json`
  (FDG ≈1014, PSMA =600; 70/10/20 within each tracer×role stratum).
- PSMA five complementary 20% test folds; FDG TEST is the single 20% source split.
- Few-shot labeled subsets: `fs50` / `fs10` / `fs5` (+ `fs0` = FDG ckpt, no PSMA FT).

**Not redistributed:** raw NIfTI volumes or checkpoints.

## Parameter alignment (shared envelope)

| Stage | Global batch | Epochs | Train iters / epoch |
|-------|-------------:|-------:|--------------------:|
| FDG supervised (nnU-Net-style: nnU-Net, DpDNet, TSACL) | 6 | 169 | 70 |
| FDG supervised (transformer/foundation: SwinViT, PET/CT MAE, SegAnyPET) | 6 | 100 | 70 (when used) |
| PSMA `fs50` / `fs10` / `fs5` (all methods) | 2 | 100 | 25 |
| PSMA `fs0` | — | — | FDG checkpoint only |
| PSMA `ss70` / FDG TEST | ref. columns | single-split | — |

Scratch twin ↔ pretrained sibling share the same batch/epoch/iters for that row.
Primary degree of freedom on the board is **Init**, not optimizer shopping.

## Methods (one line each)

| Method | Role on the board |
|--------|-------------------|
| nnU-Net | Self-configuring CNN U-Net toolkit (MIM vs scratch) |
| SwinViT | Swin UNETR + Tang et al. 3D medical SSL |
| DpDNet | Dual-prompt PET/CT segmenter (dual-enc vs scratch) |
| SegAnyPET | Promptable PET segmenter; scored **fully automatic** (no clicks) |
| PET/CT MAE | 3D MAE foundation pretraining on PET/CT |
| TSACL | AutoPET~V interactive winner; automatic board reference |

## Launchers

Background scripts live under `run/` (source `run/_aligned_seed_env.sh`).
Board JSON / live dashboard: see upstream `ICLR2026/vis/` when developing locally.

## AAAI / review note

AAAI-27 forbids **web links** to code or anonymous repos in the submission PDF
(including Anonymous GitHub). For review, ship a **supplementary code ZIP**;
point camera-ready readers to this repository after acceptance.
