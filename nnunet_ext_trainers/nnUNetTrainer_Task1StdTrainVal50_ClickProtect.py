"""
Task1StdTrainVal50 + 点击/级联通道强度 DA 保护。

环境变量:
  TASK1_OVERSAMPLE_FOREGROUND_PERCENT   设则覆盖 oversample（lowres 常用 0.33）
  TASK1_PROTECT_INTENSITY_DA            1/0，默认 1
  TASK1_PROTECT_INTENSITY_DA_CHANNELS   逗号分隔通道 index，如 2,3 或 2,3,4
      lowres 默认 2,3（FG/BG 高斯点击）
      cascade（is_cascaded）默认 2,3,4（+ lowres seg 第 5 通道输入）
"""
from __future__ import annotations

import os
from typing import List, Tuple, Union

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

from nnunet_ext_trainers.task1_intensity_da_protect import (
    parse_protect_channel_list,
    wrap_intensity_da_protect_channels,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as _nnUNetTrainerBase
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_Task1StdTrainVal50 import (
    nnUNetTrainer_Task1StdTrainVal50,
)


def _env_truthy(name: str, default: str = "1") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class nnUNetTrainer_Task1StdTrainVal50_ClickProtect(nnUNetTrainer_Task1StdTrainVal50):
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
        v_os = os.environ.get("TASK1_OVERSAMPLE_FOREGROUND_PERCENT")
        if v_os is not None and str(v_os).strip() != "":
            self.oversample_foreground_percent = float(str(v_os).strip())
        self._protect_da = _env_truthy("TASK1_PROTECT_INTENSITY_DA", "1")
        if self.local_rank == 0:
            self.print_to_log_file(
                "[Task1StdTrainVal50_ClickProtect] "
                f"oversample_fg={self.oversample_foreground_percent} "
                f"protect_da={self._protect_da} "
                f"protect_channels_env={os.environ.get('TASK1_PROTECT_INTENSITY_DA_CHANNELS', '<auto>')}"
            )

    @staticmethod
    def _default_protect_channels(is_cascaded: bool) -> Tuple[int, ...]:
        return (2, 3, 4) if is_cascaded else (2, 3)

    @staticmethod
    def _num_image_channels(is_cascaded: bool) -> int:
        return 5 if is_cascaded else 4

    @staticmethod
    def _maybe_protect(
        composed: BasicTransform,
        is_cascaded: bool,
    ) -> BasicTransform:
        if not _env_truthy("TASK1_PROTECT_INTENSITY_DA", "1"):
            return composed
        if not isinstance(composed, ComposeTransforms):
            return composed
        protect = parse_protect_channel_list(
            os.environ.get("TASK1_PROTECT_INTENSITY_DA_CHANNELS", ""),
            nnUNetTrainer_Task1StdTrainVal50_ClickProtect._default_protect_channels(
                is_cascaded
            ),
        )
        return wrap_intensity_da_protect_channels(
            composed,
            protect,
            nnUNetTrainer_Task1StdTrainVal50_ClickProtect._num_image_channels(is_cascaded),
        )

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
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )
        return nnUNetTrainer_Task1StdTrainVal50_ClickProtect._maybe_protect(
            composed, is_cascaded
        )
