"""Production LocalEdit inference for HoleGuard Fusion v5."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from nnunetv2.inference.autopet_predictor import autoPETPredictor
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.training.dataloading.click_local_edit import (
    build_local_support,
    decide_local_edits,
)
from nnunetv2.training.dataloading.utils import sparse_to_dense_point_nnInteractive


LOCAL_EDIT_RADIUS = 12.0
LOCAL_EDIT_PROBABILITY_THRESHOLD = 0.8
LOCAL_EDIT_KEEP_MARGIN = 0.1
LOCAL_EDIT_MAX_ACTION_VOXELS = 2000
POINT_WIDTH = 2


@dataclass(frozen=True)
class LocalEditNativeResult:
    mask: np.ndarray
    add: np.ndarray
    remove: np.ndarray
    rejected_oversize_components: int
    folds: tuple[int, ...]
    cuda_peak_allocated_bytes: int
    cuda_peak_reserved_bytes: int


def configured_localedit_folds() -> tuple[int, ...]:
    raw = os.environ.get("AUTOPET_LOCAL_EDIT_FOLDS", "0,1,2,3,4")
    try:
        folds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise ValueError(f"invalid AUTOPET_LOCAL_EDIT_FOLDS={raw!r}") from error
    if not folds or any(fold not in range(5) for fold in folds) or len(set(folds)) != len(folds):
        raise ValueError(f"invalid AUTOPET_LOCAL_EDIT_FOLDS={raw!r}")
    return folds


def _inverse_preprocessed_mask(predictor, mask, data_properties):
    configuration = predictor.configuration_manager
    plans = predictor.plans_manager
    spacing_transposed = [
        data_properties["spacing"][index] for index in plans.transpose_forward
    ]
    current_spacing = (
        configuration.spacing
        if len(configuration.spacing)
        == len(data_properties["shape_after_cropping_and_before_resampling"])
        else [spacing_transposed[0], *configuration.spacing]
    )
    cropped = configuration.resampling_fn_seg(
        mask[None].astype(np.uint8, copy=False),
        data_properties["shape_after_cropping_and_before_resampling"],
        current_spacing,
        spacing_transposed,
    )[0]
    restored = np.zeros(data_properties["shape_before_cropping"], dtype=np.uint8)
    bbox = data_properties["bbox_used_for_cropping"]
    restored[tuple(slice(int(lo), int(hi)) for lo, hi in bbox)] = cropped
    return restored.transpose(plans.transpose_backward)


def _resolve_model_dir(domain: str) -> str:
    normalized = str(domain).lower().rstrip("_")
    if normalized not in ("fdg", "psma"):
        raise ValueError(f"unsupported tracer domain {domain!r}")
    configured = os.environ.get(f"AUTOPET_LOCAL_EDIT_{normalized.upper()}_MODEL")
    candidates = tuple(path for path in (
        configured,
        f"/opt/app/localedit_{normalized}",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f"localedit_{normalized}"),
    ) if path)
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "dataset.json")) and os.path.isfile(
            os.path.join(candidate, "plans.json")
        ):
            return candidate
    raise FileNotFoundError(f"LocalEdit {normalized} model was not baked into the image")


def predict_localedit_native(img, props, champion_base, clicks, domain):
    """Return the frozen LocalEdit candidate reconstructed from immutable M0."""
    folds = configured_localedit_folds()
    memory_limit_gib = os.environ.get("AUTOPET_CUDA_MEMORY_LIMIT_GIB")
    if memory_limit_gib:
        requested_bytes = float(memory_limit_gib) * 1024**3
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        if not 0 < requested_bytes <= total_bytes:
            raise ValueError(
                f"invalid AUTOPET_CUDA_MEMORY_LIMIT_GIB={memory_limit_gib!r} "
                f"for device bytes={total_bytes}"
            )
        torch.cuda.set_per_process_memory_fraction(requested_bytes / total_bytes, 0)
    torch.cuda.reset_peak_memory_stats(0)
    predictor = autoPETPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        _resolve_model_dir(domain),
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )
    original_heads = predictor.label_manager.num_segmentation_heads
    if original_heads != 2:
        raise RuntimeError(
            f"LocalEdit expected source dataset to declare two labels, got {original_heads}"
        )
    predictor.label_manager = SimpleNamespace(num_segmentation_heads=3)

    champion_base = (np.asarray(champion_base) > 0).astype(np.uint8)
    if champion_base.shape != tuple(img.shape[1:]):
        raise RuntimeError(
            f"champion/input shape mismatch {champion_base.shape} != {tuple(img.shape[1:])}"
        )
    adapter = PreprocessAdapterFromNpy(
        [img],
        # nnU-Net writes its nonzero-crop sentinel (-1) into the previous-stage
        # segmentation. NumPy 2 correctly rejects that assignment for uint8;
        # signed int16 preserves the identical 0/1 M0 while allowing the
        # temporary sentinel outside the crop.
        [champion_base[None].astype(np.int16, copy=False)],
        [props],
        [None],
        predictor.plans_manager,
        predictor.dataset_json,
        predictor.configuration_manager,
        num_threads_in_multithreaded=1,
        verbose=False,
    )
    preprocessed = next(adapter)
    model_data = preprocessed["data"]
    if model_data.shape[0] != 3:
        raise RuntimeError(f"LocalEdit preprocessing expected CT/PET/M0, got {tuple(model_data.shape)}")

    positive, negative = sparse_to_dense_point_nnInteractive(
        clicks.get("points", []),
        model_data.shape[1:],
        preprocessed["data_properties"],
        sigma=POINT_WIDTH,
    )
    click_channels = torch.stack((positive, negative))
    click_cores_np = click_channels.numpy() >= 0.999
    support_np = build_local_support(click_cores_np, radius_voxels=LOCAL_EDIT_RADIUS)
    support_channels = torch.from_numpy(support_np).float()
    seven_channel = torch.vstack((model_data, click_channels, support_channels))
    if tuple(seven_channel.shape[:1]) != (7,):
        raise RuntimeError(f"LocalEdit expected seven channels, got {tuple(seven_channel.shape)}")

    logits = predictor.predict_logits_from_preprocessed_data(seven_channel).float().cpu()
    if tuple(logits.shape[:1]) != (3,):
        raise RuntimeError(f"LocalEdit expected three action logits, got {tuple(logits.shape)}")
    champion = model_data[2:3] > 0.5
    decisions = decide_local_edits(
        logits[None],
        champion[None],
        torch.from_numpy(click_cores_np)[None],
        torch.from_numpy(support_np)[None],
        probability_threshold=LOCAL_EDIT_PROBABILITY_THRESHOLD,
        keep_margin=LOCAL_EDIT_KEEP_MARGIN,
        max_action_voxels=LOCAL_EDIT_MAX_ACTION_VOXELS,
    )
    if len(decisions) != 1:
        raise RuntimeError(f"LocalEdit expected one decision, got {len(decisions)}")
    decision = decisions[0]
    native_add = _inverse_preprocessed_mask(
        predictor, decision.add.astype(np.uint8), preprocessed["data_properties"]
    ) > 0
    native_remove = _inverse_preprocessed_mask(
        predictor, decision.remove.astype(np.uint8), preprocessed["data_properties"]
    ) > 0
    local_native = ((champion_base > 0) | native_add) & ~native_remove
    if not np.any(decision.add) and not np.any(decision.remove):
        if not np.array_equal(local_native, champion_base > 0):
            changed = int(np.count_nonzero(local_native ^ (champion_base > 0)))
            raise RuntimeError(f"LocalEdit zero-action identity violated by {changed} voxels")

    return LocalEditNativeResult(
        mask=local_native,
        add=native_add,
        remove=native_remove,
        rejected_oversize_components=int(decision.rejected_oversize_components),
        folds=folds,
        cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
        cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
    )
