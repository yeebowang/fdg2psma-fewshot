# AutoPET V 2026 — LocalEdit–TACE HoleGuard Fusion V5

This repository builds the Grand Challenge algorithm container used for the
AutoPET V 2026 Final Test phase. The method performs interactive whole-body
PET/CT lesion segmentation for both FDG and PSMA studies.

## Method

The initial prediction (`iter0`) is the five-fold AutoPET-III LesionTracer
output with a frozen tracer-specific connected-component policy. FDG keeps
18-connected components of at least 25 voxels. PSMA keeps 18-connected
components of at least five voxels and uses a frozen three-feature logistic
gate to recover selected 1--4 voxel components. The gate uses component size,
physical volume, and bounding-box fill fraction; its coefficients and scaler
are versioned in
[`psma_shape_component_gate_final.json`](psma_shape_component_gate_final.json).
Each later invocation receives all cumulative tumor/background clicks and
combines two frozen correction branches:

The FDG/PSMA route is selected directly from the input PET/CT by a fixed
three-classifier majority vote (histogram gradient boosting, linear SVM, and
random forest). Its 99 robust whole-body distribution features sanitize
non-finite or implausible intensities before inference. The router achieved
1,607/1,607 correct patient-grouped out-of-fold decisions in the released
training cohort; a disagreement flag is emitted whenever the three classifiers
do not vote unanimously.

For a zero-click invocation, the same five-fold champion ensemble writes its
hard mask without exporting an unused full-volume probability archive, and the
parent process releases the routing arrays before inference. On every
invocation the champion child finishes before the parent imports Torch or
initializes CUDA; interactive invocations then export the float32 probability
map required by TACE and load the correction stack. This preserves a clean
champion GPU context while increasing host-memory and temporary-storage
headroom on large whole-body scans.

1. **TACE (Topology-Aware Click Editing)** edits the champion probability map
   locally using tracer-specific add/remove responses and topology checks.
2. **LocalEdit** is a five-fold FDG or PSMA specialist. Its seven input channels
   are CT, PET, immutable champion M0, positive/negative click maps, and two
   local-support maps. It predicts KEEP/ADD/REMOVE actions.

The V5 transaction gate applies both proposals to the previous accepted mask.
REMOVE transactions that split a previous component are rejected. HoleGuard
falls back first to TACE and then to the previous mask if a proposal creates a
new enclosed hole. This release uses the frozen V5 transaction semantics. In
the strict 1,607-case OOF comparison, the final component policy improved
AUC-Dice by 0.000251 (95% CI 0.000093--0.000437) and AUC-DMM by 0.003549
(95% CI 0.002030--0.005109) over the preceding V5 container policy. Technical
telemetry keeps the historical `gaussian_v6` identifier so frozen evaluation
artifacts remain traceable; the public method name is TACE.

## Frozen weights

The Docker build downloads the ten slim LocalEdit checkpoints from the public
`weights-v0.3.0` release. Each domain archive is carried as ordered multipart
assets (`chunk-aa` through `chunk-bt`) to make the large release reliably
reproducible. The build joins them byte-for-byte and verifies the canonical
archive SHA-256 before extraction:

| Domain | Archive | SHA-256 |
| --- | --- | --- |
| FDG | `localedit_tace_holeguard_v5_fdg.tgz` | `08b18f819b8fecc7f19f48bc84f5d09610b9762128169a83d9802e68fe865f62` |
| PSMA | `localedit_tace_holeguard_v5_psma.tgz` | `b4c435063dfff7469cfce76d98fd7971307542453f1c8caadb108f0b5d632f5d` |

The AutoPET-III champion weights are downloaded from their original Zenodo
record during the image build and verified against the record's published MD5
(`566016409b0bd14770c0b57c1f2873f1`, 3,808,128,600 bytes). No model download
occurs during inference. Per-fold checkpoint byte sizes, SHA-256 hashes, and
network-tensor hashes are recorded in
[`WEIGHTS_MANIFEST.json`](WEIGHTS_MANIFEST.json).
That manifest also records every part's byte size, SHA-256, and join order; the
parts are transport packaging only and do not change checkpoint tensors.

## Build

```bash
docker build --platform=linux/amd64 \
  -t autopet-v-v5:final .
```

The PyTorch 2.6.0/CUDA 12.4 base image is pinned by OCI digest. Direct Python
dependencies are locked in [`constraints.txt`](constraints.txt) to the
versions used by the qualified T4 runtime, including NumPy 2.1.2, SciPy 1.17.1,
and SimpleITK 2.5.5.

The final Grand Challenge runtime has no network access. All champion and
LocalEdit weights are baked into the image by the build above.

## Container interface

The container follows the official AutoPET V sockets:

```text
/input/images/ct/<case>.mha
/input/images/pet/<case>.mha
/input/lesion-clicks.json
/output/images/tumor-lesion-segmentation/<case>.mha
```

`lesion-clicks.json` contains cumulative `tumor` and `background` points. The
output geometry is copied from the input CT. Interactive calls load the previous
accepted output when available and fail closed to M0 if that state is missing or
invalid.

## Qualification policy

The release candidate is accepted only after all of the following pass:

- strict five-fold OOF coverage over all 1,607 training studies;
- FDG and PSMA six-step container replays with iter0 bitwise identity;
- no newly accepted enclosed holes, plus a 24-case structural and human visual
  review of accepted edits;
- ten packaged checkpoint hashes matching the audited slim-weight manifest;
- container source hashes matching this repository;
- peak reserved CUDA memory below 16 GiB and each invocation below 900 seconds.

The internal OOF and clinical-image audit artifacts are intentionally not
included in this source repository because they contain dataset-derived case
identifiers and predictions.

## License

The submission source is released under the Apache License 2.0; see
[`LICENSE`](LICENSE). Bundled AutoPET-III LesionTracer and
`autoPET-interactive` source components retain their upstream Apache 2.0
license files and attribution.
