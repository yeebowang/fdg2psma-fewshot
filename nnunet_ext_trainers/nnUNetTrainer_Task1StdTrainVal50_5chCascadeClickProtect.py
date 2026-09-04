"""
[3.5/5] 五通道 fullres + cascade fullres seg one-hot + ClickProtect 强度 DA。

在 5chCascadeProb 基础上，用 TASK1_PROTECT_INTENSITY_DA_CHANNELS（默认 2,3,4）
保护点击通道与 cascade seg 通道，避免强度 DA 破坏旁路输入。

环境变量:
  TASK1_PROTECT_INTENSITY_DA            1/0，默认 1
  TASK1_PROTECT_INTENSITY_DA_CHANNELS   默认 2,3,4
  TASK1_5CH_SPLIT_CASCADE_SEG           同 5chCascadeProb
"""
from __future__ import annotations

import os
from typing import Tuple, Union

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

from nnunet_ext_trainers.nnUNetTrainer_Task1StdTrainVal50_5chCascadeProb import (
    nnUNetTrainer_Task1StdTrainVal50_5chCascadeProb,
)
from nnunet_ext_trainers.task1_intensity_da_protect import (
    parse_protect_channel_list,
    wrap_intensity_da_protect_channels,
)


def _env_truthy(name: str, default: str = "1") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class nnUNetTrainer_Task1StdTrainVal50_5chCascadeClickProtect(
    nnUNetTrainer_Task1StdTrainVal50_5chCascadeProb
):
    """5ch cascade seg + ClickProtect（ch2/3/4 强度 DA 保护）。"""

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
        if self.local_rank == 0:
            self.print_to_log_file(
                "[Task1StdTrainVal50_5chCascadeClickProtect] "
                f"protect_da={_env_truthy('TASK1_PROTECT_INTENSITY_DA', '1')} "
                f"protect_channels={os.environ.get('TASK1_PROTECT_INTENSITY_DA_CHANNELS', '2,3,4')}"
            )

    @staticmethod
    def _default_protect_channels() -> Tuple[int, ...]:
        return (2, 3, 4)

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA,
        deep_supervision_scales: Union[list, tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: list[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], list[int]] = None,
        regions: list[Union[list[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        composed = nnUNetTrainer_Task1StdTrainVal50_5chCascadeProb.get_training_transforms(
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
        if not _env_truthy("TASK1_PROTECT_INTENSITY_DA", "1"):
            return composed
        if not isinstance(composed, ComposeTransforms):
            return composed
        protect = parse_protect_channel_list(
            os.environ.get("TASK1_PROTECT_INTENSITY_DA_CHANNELS", ""),
            nnUNetTrainer_Task1StdTrainVal50_5chCascadeClickProtect._default_protect_channels(),
        )
        return wrap_intensity_da_protect_channels(composed, protect, num_channels=5)
