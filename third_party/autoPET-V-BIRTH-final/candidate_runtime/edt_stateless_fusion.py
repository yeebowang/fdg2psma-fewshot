"""Pure stateless component fusion used by the deployable EDT candidate."""

from collections.abc import Mapping, Sequence

import cc3d
import numpy as np


def _valid_coordinates(
    values: Sequence, shape: tuple[int, int, int]
) -> np.ndarray:
    result = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            continue
        try:
            coordinate = [int(item) for item in value]
        except (TypeError, ValueError):
            continue
        if all(0 <= coordinate[axis] < shape[axis] for axis in range(3)):
            result.append(coordinate)
    return np.asarray(result, dtype=np.int64).reshape(-1, 3)


def _component_count(mask: np.ndarray) -> int:
    _, count = cc3d.connected_components(
        np.asarray(mask, dtype=np.uint8), connectivity=18, return_N=True
    )
    return int(count)


def fuse_clicked_components(
    initial: np.ndarray,
    edt_prediction: np.ndarray,
    scribbles: Mapping[str, Sequence],
    tracer: str,
    *,
    disable_background_edits: bool = False,
) -> np.ndarray:
    """Fuse cumulative clicks into k0 without relying on previous container state."""
    initial = np.asarray(initial, dtype=bool)
    donor = np.asarray(edt_prediction, dtype=bool)
    if initial.shape != donor.shape:
        raise ValueError(f"Initial/EDT grid mismatch: {initial.shape} != {donor.shape}")
    tracer = tracer.strip().lower()
    if tracer not in {"fdg", "psma"}:
        raise ValueError(f"Unsupported tracer: {tracer}")
    tumor = _valid_coordinates(scribbles.get("tumor", []), initial.shape)
    background = _valid_coordinates(scribbles.get("background", []), initial.shape)
    fused = initial.copy()
    initial_count = _component_count(initial)

    # The PSMA full gate showed that fewer than six cumulative foreground
    # points represent tiny boundary corrections that are safer to propagate.
    allow_tumor = len(tumor) > 0
    if tracer == "psma":
        allow_tumor = allow_tumor and len(tumor) >= 6 and initial_count <= 10
    if allow_tumor:
        donor_labels, _ = cc3d.connected_components(
            donor.astype(np.uint8), connectivity=18, return_N=True
        )
        index = tuple(tumor[:, axis] for axis in range(3))
        target_labels = set(
            int(value) for value in np.unique(donor_labels[index]) if value > 0
        )
        candidate = fused.copy()
        for target_label in target_labels:
            candidate |= donor_labels == target_label
        # Never bridge two lesions that were separate in the safe k0 mask.
        if _component_count(candidate) >= initial_count:
            fused = candidate

    # PSMA background edits are disabled by the zero-loss full validation.
    if tracer == "fdg" and len(background) > 0 and not disable_background_edits:
        labels, _ = cc3d.connected_components(
            fused.astype(np.uint8), connectivity=18, return_N=True
        )
        index = tuple(background[:, axis] for axis in range(3))
        target_labels = set(
            int(value) for value in np.unique(labels[index]) if value > 0
        )
        candidate = fused.copy()
        for target_label in target_labels:
            target = labels == target_label
            candidate[target] = donor[target]
        # Reject fragmentation that would create lesion-level false positives.
        if _component_count(candidate) <= _component_count(fused):
            fused = candidate
    return fused
