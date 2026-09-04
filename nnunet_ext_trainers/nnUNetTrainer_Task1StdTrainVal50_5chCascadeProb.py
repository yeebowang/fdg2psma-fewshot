"""
[3.5/5] 五通道 fullres：CT+PET+FG_CLICK+BG_CLICK+CASCADE_SEG（cascade fullres 二值 one-hot）。
纯分割 loss；ch4 加载与 nnUNet cascade fullres 一致（mmap patch + MoveSegAsOneHot + cascade DA），
仅旁路 seg 来自 cascade fullres 而非 lowres。

环境变量:
  TASK1_5CH_SPLIT_CASCADE_SEG         1/0，默认 1（兼容 TASK1_5CH_SPLIT_PROB）
  TASK1_5CH_CASCADE_SEG_B2ND_DIR      默认 {prep}/cascade_fullres_seg_uint8
  TASK1_5CH_PROTECT_PROB_DA           1/0，默认 1（保护 cascade one-hot 通道）
  TASK1_5CH_PROTECT_PROB_DA_CHANNEL   默认 4
"""

from __future__ import annotations

import os
from typing import List, Tuple, Union

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import (
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from nnunet_ext_trainers.task1_3ch_protect_seg_da import (
    RestoreImageChannelsTransform,
    SaveImageChannelsTransform,
)
from nnunet_ext_trainers.task1_5ch_split_prob_dataset import (
    nnUNetDatasetBlosc2_Task1_5chSplitCascadeSeg,
    task1_5ch_cascade_seg_b2nd_dir,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as _nnUNetTrainerBase
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_Task1StdTrainVal50 import (
    nnUNetTrainer_Task1StdTrainVal50,
)


def _env_truthy(name: str, default: str = "1") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(str(raw).strip())


def _split_cascade_seg_enabled() -> bool:
    if "TASK1_5CH_SPLIT_CASCADE_SEG" in os.environ:
        return _env_truthy("TASK1_5CH_SPLIT_CASCADE_SEG", "1")
    return _env_truthy("TASK1_5CH_SPLIT_PROB", "1")


class nnUNetTrainer_Task1StdTrainVal50_5chCascadeProb(nnUNetTrainer_Task1StdTrainVal50):
    """5ch fullres + cascade fullres 二值 seg one-hot 输入，无 ref loss。"""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        self._split_cascade_seg = _split_cascade_seg_enabled()
        self._protect_cascade_da = _env_truthy("TASK1_5CH_PROTECT_PROB_DA", "1")
        self._protect_cascade_da_channel = _env_int("TASK1_5CH_PROTECT_PROB_DA_CHANNEL", 4)
        if self.local_rank == 0:
            split_msg = (
                "split_cascade_seg=1 seg_dir=<preprocessed>/cascade_fullres_seg_uint8 "
                "load=mmap_patch+onehot (cascade_fullres style)"
                if self._split_cascade_seg
                else "split_cascade_seg=off"
            )
            self.print_to_log_file(
                "[Task1StdTrainVal50_5chCascadeProb] "
                f"ref_loss=0 (seg only) "
                f"protect_cascade_da={self._protect_cascade_da} "
                f"protect_channel={self._protect_cascade_da_channel} "
                f"{split_msg}"
            )

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()
        if self._split_cascade_seg:
            if self.dataset_class is None:
                from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

                self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            seg_dir = task1_5ch_cascade_seg_b2nd_dir(self.preprocessed_dataset_folder)
            if self.local_rank == 0:
                self.print_to_log_file(
                    f"[Task1StdTrainVal50_5chCascadeProb] split_cascade_seg mmap+onehot: "
                    f"4ch b2nd + uint8 seg from {seg_dir}"
                )
            cls = nnUNetDatasetBlosc2_Task1_5chSplitCascadeSeg
        else:
            cls = self.dataset_class
        dataset_tr = cls(self.preprocessed_dataset_folder, tr_keys)
        dataset_val = cls(self.preprocessed_dataset_folder, val_keys)
        return dataset_tr, dataset_val

    @staticmethod
    def _wrap_intensity_da_protect_channel(
        composed: ComposeTransforms,
        protect_channel: int,
        num_channels: int = 5,
    ) -> ComposeTransforms:
        transforms: List[BasicTransform] = list(composed.transforms)
        intensity_types = (
            GaussianNoiseTransform,
            GaussianBlurTransform,
            MultiplicativeBrightnessTransform,
            ContrastTransform,
            SimulateLowResolutionTransform,
            GammaTransform,
        )
        first_intensity = last_intensity_plus_one = None
        allowed = tuple(c for c in range(num_channels) if c != protect_channel)
        for i, tr in enumerate(transforms):
            inner = tr.transform if isinstance(tr, RandomTransform) else tr
            if isinstance(inner, intensity_types):
                if first_intensity is None:
                    first_intensity = i
                last_intensity_plus_one = i + 1
            if isinstance(inner, SimulateLowResolutionTransform):
                inner.allowed_channels = allowed
        if first_intensity is None or last_intensity_plus_one is None:
            return composed
        ch = (protect_channel,)
        wrapped = (
            transforms[:first_intensity]
            + [SaveImageChannelsTransform(ch)]
            + transforms[first_intensity:last_intensity_plus_one]
            + [RestoreImageChannelsTransform(ch)]
            + transforms[last_intensity_plus_one:]
        )
        return ComposeTransforms(wrapped)

    @staticmethod
    def _prepend_cascade_fullres_seg_transforms(
        composed: BasicTransform,
        foreground_labels: Union[Tuple[int, ...], List[int]] | None,
        training: bool,
    ) -> BasicTransform:
        if not _split_cascade_seg_enabled():
            return composed
        if not isinstance(composed, ComposeTransforms):
            return composed
        if foreground_labels is None:
            raise ValueError("foreground_labels required for cascade seg one-hot")
        fl = list(foreground_labels)
        ch_idx = list(range(-len(fl), 0))
        prelude: List[BasicTransform] = [
            MoveSegAsOneHotToDataTransform(
                source_channel_idx=1,
                all_labels=fl,
                remove_channel_from_source=True,
            ),
        ]
        if training:
            prelude.extend(
                [
                    RandomTransform(
                        ApplyRandomBinaryOperatorTransform(
                            channel_idx=ch_idx,
                            strel_size=(1, 8),
                            p_per_label=1,
                        ),
                        apply_probability=0.4,
                    ),
                    RandomTransform(
                        RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                            channel_idx=ch_idx,
                            fill_with_other_class_p=0,
                            dont_do_if_covers_more_than_x_percent=0.15,
                            p_per_label=1,
                        ),
                        apply_probability=0.2,
                    ),
                ]
            )
        return ComposeTransforms(prelude + list(composed.transforms))

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        composed = _nnUNetTrainerBase.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=False,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )
        composed = nnUNetTrainer_Task1StdTrainVal50_5chCascadeProb._prepend_cascade_fullres_seg_transforms(
            composed, foreground_labels, training=True
        )
        if not isinstance(composed, ComposeTransforms):
            return composed
        if _env_truthy("TASK1_5CH_PROTECT_PROB_DA", "1"):
            protect_ch = _env_int("TASK1_5CH_PROTECT_PROB_DA_CHANNEL", 4)
            composed = nnUNetTrainer_Task1StdTrainVal50_5chCascadeProb._wrap_intensity_da_protect_channel(
                composed, protect_ch, num_channels=5
            )
        return composed

    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales: Union[List, Tuple, None],
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        composed = _nnUNetTrainerBase.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=False,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )
        return nnUNetTrainer_Task1StdTrainVal50_5chCascadeProb._prepend_cascade_fullres_seg_transforms(
            composed, foreground_labels, training=False
        )
