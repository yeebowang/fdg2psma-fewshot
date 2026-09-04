# AutoPET V BIRTH Final Submission

Reproducible source for the BIRTH Lab AutoPET V 2026 submission. The method
combines a five-fold AutoPET-III LesionTracer initial prediction with
tracer-specific K0 calibration and stateless EDT click correction.

## Method

- A fixed three-model router classifies FDG versus PSMA from PET/CT.
- K0 uses the five-fold AutoPET-III LesionTracer model with mirror TTA disabled.
- FDG uses probability threshold `0.47`; high-burden scans (at least 25 robust
  components) relax connected-component dust from 25 to 5 voxels.
- PSMA uses threshold `0.50`, dust 5, and a frozen logistic component pruner at
  false-positive threshold `0.86`.
- Interactive calls run the AutoPET-IV LesionLocator EDT fold-0 checkpoint.
  Tumor corrections are component-local and reject lesion bridges. Background
  deletion is disabled. PSMA correction requires at least six cumulative tumor
  points and at most ten K0 components.
- Each invocation is stateless: K0 is reconstructed and all cumulative clicks
  are replayed.

The Grand Challenge interface is:

```text
/input/images/ct/<case>.mha
/input/images/pet/<case>.mha
/input/lesion-clicks.json
/output/images/tumor-lesion-segmentation/<case>.mha
```

## Reproducible build

The Docker build downloads the public champion weights from Zenodo and the EDT
checkpoint from this repository's `weights-v1.0.0` GitHub Release. Both are
verified before use.

```bash
docker build --platform=linux/amd64 -t autopet-v-birth:final .
```

Frozen EDT checkpoint SHA-256:

```text
a0cb3a89c72b0a79a27900980361385ff02572c0c71aba6609390fecbbc13e82
```

No network access is used during inference; all weights are baked into the
image at build time.

## Model checkpoints

| Component | Source | Integrity check |
| --- | --- | --- |
| AutoPET-III LesionTracer, folds 0–4 | [Zenodo 14007247](https://zenodo.org/records/14007247) | MD5 `566016409b0bd14770c0b57c1f2873f1` |
| LesionLocator EDT, fold 0 | GitHub Release `weights-v1.0.0` | SHA-256 `a0cb3a89c72b0a79a27900980361385ff02572c0c71aba6609390fecbbc13e82` |

The 820 MB EDT checkpoint is intentionally not committed to Git. GitHub's
per-file source limit is 100 MB, so the checkpoint is distributed as a
versioned Release asset and verified during the Docker build.

## Repository layout

```text
candidate_runtime/       Final K0, PSMA pruning, and stateless fusion
autoPET-interactive/     Pinned EDT/nnU-Net fork (Apache-2.0)
champion/                Pinned AutoPET-III inference fork (Apache-2.0)
weights/edt_model/       EDT plans, metadata, and expected checkpoint hash
tests/                   Runtime and safety-gate unit tests
Dockerfile               Reproducible, digest-pinned container build
edt_runner.py             Isolated fold-0 EDT inference process
public_tracer_router.py   Fixed FDG/PSMA router
```

## Tests

```bash
python -m pytest -q
```

The tests cover invalid inputs, PSMA component pruning, cumulative-click
activation, topology-preserving foreground edits, and disabled background
deletion.

## Validation summary

On the Preliminary Test Set, the submitted K0 obtained Dice `0.853649` and
lesion F1 `0.822220`. Local patient-independent validation was used for all
K0 and interaction decisions. The final ranking itself is computed by the
challenge organizers using AUC-Dice and AUC-DMM over six interaction states.

## Provenance

- AutoPET V public implementation: bundled Apache-2.0 source snapshot.
- AutoPET-III LesionTracer source and weights: bundled upstream code and Zenodo
  record 14007247.
- AutoPET interactive/EDT source: `MIC-DKFZ/autoPET-interactive`, pinned commit
  `0da0e7f`.

See `NOTICE` for attribution and modification notes.

## Citation

The challenge method-description preprint will be linked here once its public
identifier is available. Until then, please cite the upstream AutoPET-III and
AutoPET interactive projects listed above when reusing their components.

## License

Apache License 2.0. Bundled third-party source retains its upstream notices.
