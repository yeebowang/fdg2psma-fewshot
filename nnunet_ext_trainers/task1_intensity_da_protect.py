"""强度类 DA 前后保护指定 image 通道（空间 DA 仍同步）。"""
from __future__ import annotations

from typing import List, Tuple

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from nnunet_ext_trainers.task1_3ch_protect_seg_da import (
    RestoreImageChannelsTransform,
    SaveImageChannelsTransform,
)

_INTENSITY_TYPES = (
    GaussianNoiseTransform,
    GaussianBlurTransform,
    MultiplicativeBrightnessTransform,
    ContrastTransform,
    SimulateLowResolutionTransform,
    GammaTransform,
)


def parse_protect_channel_list(raw: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    text = (raw or "").strip()
    if not text:
        return default
    out: List[int] = []
    for part in text.replace(" ", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return tuple(out) if out else default


def wrap_intensity_da_protect_channels(
    composed: ComposeTransforms,
    protect_channels: Tuple[int, ...],
    num_channels: int,
) -> ComposeTransforms:
    if not protect_channels:
        return composed
    transforms: List[BasicTransform] = list(composed.transforms)
    protect_set = set(protect_channels)
    allowed = tuple(c for c in range(num_channels) if c not in protect_set)
    first_intensity = last_intensity_plus_one = None
    for i, tr in enumerate(transforms):
        inner = tr.transform if isinstance(tr, RandomTransform) else tr
        if isinstance(inner, _INTENSITY_TYPES):
            if first_intensity is None:
                first_intensity = i
            last_intensity_plus_one = i + 1
        if isinstance(inner, SimulateLowResolutionTransform):
            inner.allowed_channels = allowed
    if first_intensity is None or last_intensity_plus_one is None:
        return composed
    wrapped = (
        transforms[:first_intensity]
        + [SaveImageChannelsTransform(protect_channels)]
        + transforms[first_intensity:last_intensity_plus_one]
        + [RestoreImageChannelsTransform(protect_channels)]
        + transforms[last_intensity_plus_one:]
    )
    return ComposeTransforms(wrapped)
