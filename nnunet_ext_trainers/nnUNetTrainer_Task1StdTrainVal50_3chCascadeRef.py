"""
[3.5/5] 三通道 fullres：在 Task1StdTrainVal50 基础上，增加输出与输入第三通道（CASCADE_SEG / low1933+cas2936）的差异项，
约束 3.5 分割结果靠近 cascade 伪标签。

总 loss（默认归一化）:
  L = L_seg/|L_seg| + TASK1_3CH_CASCADE_REF_WEIGHT * L_ref/|L_ref|
  其中 |·| 为 detach 绝对值，下限 clamp 为 1.0（与 Task1MambaKD 一致）。

L_ref 默认 BCEWithLogits(前景 logit, binary ref)；第三通道为 0/1 cascade 伪标签（NoNormalization）。

环境变量:
  TASK1_3CH_CASCADE_REF_WEIGHT      默认 0.333；0 关闭
  TASK1_3CH_CASCADE_REF_MODE        bce | mse，默认 bce（二值 ref 推荐 bce）
  TASK1_3CH_CASCADE_REF_BINARY      1/0，默认 1：ref 通道按 threshold 二值化后再算 loss
  TASK1_3CH_CASCADE_REF_THRESHOLD   默认 0.5，与 val Dice / cascade predict binary 一致
  TASK1_3CH_CASCADE_REF_CHANNEL     参考通道 index，默认 2
  TASK1_3CH_CASCADE_REF_LOSS_NORM   1/0，默认 1：L_seg 与 L_ref 分别按 detach 量级归一后再加权
  TASK1_3CH_PROTECT_SEG_DA          1/0，默认 1：强度类 DA 不改动 ref 通道（默认 ch2 二值 seg）；空间 DA 仍同步三通道
  TASK1_3CH_PROTECT_SEG_DA_CHANNEL  默认 2
  TASK1_3CH_SPLIT_REF               1/0，默认 1：主 b2nd 仅 CT+PET，ch2 从 TASK1_3CH_REF_B2ND_DIR 读 uint8（同 cascade ch4）
  TASK1_3CH_REF_B2ND_DIR            默认 {prep}/nnUNetPlans_3d_fullres/cascade_seg_ref_uint8
"""

from __future__ import annotations

import os
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from torch import autocast

from nnunet_ext_trainers.task1_3ch_protect_seg_da import (
    RestoreImageChannelsTransform,
    SaveImageChannelsTransform,
)
from nnunet_ext_trainers.task1_3ch_split_ref_dataset import (
    nnUNetDatasetBlosc2_Task1_3chSplitRef,
    task1_3ch_ref_b2nd_dir,
)
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as _nnUNetTrainerBase
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_Task1StdTrainVal50 import (
    nnUNetTrainer_Task1StdTrainVal50,
)
from nnunetv2.utilities.helpers import dummy_context


def _env_truthy(name: str, default: str = "1") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(str(raw).strip())


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(str(raw).strip())


class nnUNetTrainer_Task1StdTrainVal50_3chCascadeRef(nnUNetTrainer_Task1StdTrainVal50):
    """3ch fullres + cascade 第三通道参考 loss。"""

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
        self._cascade_ref_weight = _env_float("TASK1_3CH_CASCADE_REF_WEIGHT", 0.333)
        self._cascade_ref_mode = (
            os.environ.get("TASK1_3CH_CASCADE_REF_MODE", "bce").strip().lower() or "bce"
        )
        self._cascade_ref_channel = _env_int("TASK1_3CH_CASCADE_REF_CHANNEL", 2)
        self._cascade_ref_threshold = _env_float("TASK1_3CH_CASCADE_REF_THRESHOLD", 0.5)
        self._cascade_ref_binary = _env_truthy("TASK1_3CH_CASCADE_REF_BINARY", "1")
        self._cascade_ref_loss_norm = _env_truthy("TASK1_3CH_CASCADE_REF_LOSS_NORM", "1")
        self._protect_seg_da = _env_truthy("TASK1_3CH_PROTECT_SEG_DA", "1")
        self._protect_seg_da_channel = _env_int("TASK1_3CH_PROTECT_SEG_DA_CHANNEL", 2)
        self._split_ref = _env_truthy("TASK1_3CH_SPLIT_REF", "1")
        if self.local_rank == 0:
            split_msg = (
                f"split_ref=1 ref_dir=<preprocessed>/cascade_seg_ref_uint8"
                if self._split_ref
                else "split_ref=off"
            )
            self.print_to_log_file(
                "[Task1StdTrainVal50_3chCascadeRef] "
                f"ref_weight={self._cascade_ref_weight} "
                f"ref_mode={self._cascade_ref_mode} "
                f"ref_binary={self._cascade_ref_binary} "
                f"ref_threshold={self._cascade_ref_threshold} "
                f"ref_channel={self._cascade_ref_channel} "
                f"ref_loss_norm={self._cascade_ref_loss_norm} "
                f"protect_seg_da={self._protect_seg_da} "
                f"protect_seg_da_channel={self._protect_seg_da_channel} "
                f"{split_msg}"
            )

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()
        if self._split_ref:
            if self.dataset_class is None:
                from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

                self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            ref_dir = task1_3ch_ref_b2nd_dir(self.preprocessed_dataset_folder)
            if self.local_rank == 0:
                self.print_to_log_file(
                    f"[Task1StdTrainVal50_3chCascadeRef] split_ref load: "
                    f"2ch b2nd + uint8 ref from {ref_dir}"
                )
            cls = nnUNetDatasetBlosc2_Task1_3chSplitRef
        else:
            cls = self.dataset_class
        dataset_tr = cls(self.preprocessed_dataset_folder, tr_keys)
        dataset_val = cls(self.preprocessed_dataset_folder, val_keys)
        return dataset_tr, dataset_val

    @staticmethod
    def _wrap_intensity_da_protect_seg(
        composed: ComposeTransforms,
        protect_channel: int,
    ) -> ComposeTransforms:
        """强度类 DA 前后暂存/恢复指定通道；空间 DA（旋转/缩放/镜像）仍作用于全 patch。"""
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
        for i, tr in enumerate(transforms):
            inner = tr.transform if isinstance(tr, RandomTransform) else tr
            if isinstance(inner, intensity_types):
                if first_intensity is None:
                    first_intensity = i
                last_intensity_plus_one = i + 1
            if isinstance(inner, SimulateLowResolutionTransform):
                inner.allowed_channels = (0, 1) if protect_channel == 2 else tuple(
                    c for c in range(3) if c != protect_channel
                )
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
        if not _env_truthy("TASK1_3CH_PROTECT_SEG_DA", "1"):
            return composed
        if not isinstance(composed, ComposeTransforms):
            return composed
        protect_ch = _env_int("TASK1_3CH_PROTECT_SEG_DA_CHANNEL", 2)
        return nnUNetTrainer_Task1StdTrainVal50_3chCascadeRef._wrap_intensity_da_protect_seg(
            composed, protect_ch
        )

    def _foreground_logits(self, output: torch.Tensor | list | tuple) -> torch.Tensor:
        logits = output[0] if isinstance(output, (list, tuple)) else output
        if self.label_manager.has_regions:
            return logits
        if logits.shape[1] < 2:
            raise ValueError(f"expected >=2 output channels, got {logits.shape[1]}")
        return logits[:, 1:2]

    def _cascade_ref_loss(self, output: torch.Tensor | list | tuple, data: torch.Tensor) -> torch.Tensor:
        if self._cascade_ref_weight <= 0.0:
            return data.new_zeros(())
        ch = self._cascade_ref_channel
        if data.shape[1] <= ch:
            return data.new_zeros(())
        ref = data[:, ch : ch + 1].float().clamp(0.0, 1.0)
        if self._cascade_ref_binary:
            ref = (ref >= self._cascade_ref_threshold).float()
        fg_logits = self._foreground_logits(output)
        if fg_logits.shape[2:] != ref.shape[2:]:
            ref = F.interpolate(ref, size=fg_logits.shape[2:], mode="nearest")
        if self._cascade_ref_mode == "mse" and not self._cascade_ref_binary:
            pred_prob = torch.sigmoid(fg_logits)
            return F.mse_loss(pred_prob, ref)
        return F.binary_cross_entropy_with_logits(fg_logits, ref)

    def _combine_loss(
        self,
        output: torch.Tensor | list | tuple,
        target,
        data: torch.Tensor,
    ) -> torch.Tensor:
        l_seg = self.loss(output, target)
        l_ref = self._cascade_ref_loss(output, data)
        use_ref = self._cascade_ref_weight > 0.0 and float(l_ref.detach()) != 0.0

        if not self._cascade_ref_loss_norm:
            if not use_ref:
                return l_seg
            return l_seg + self._cascade_ref_weight * l_ref

        s_seg = torch.clamp(l_seg.detach().abs(), min=1.0)
        if not use_ref:
            return l_seg / s_seg
        s_ref = torch.clamp(l_ref.detach().abs(), min=1.0)
        return (l_seg / s_seg) + self._cascade_ref_weight * (l_ref / s_ref)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        use_cuda = self.device.type == "cuda"
        with autocast(self.device.type, enabled=use_cuda) if use_cuda else dummy_context():
            output = self.network(data)
            l = self._combine_loss(output, target, data)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        use_cuda = self.device.type == "cuda"
        with autocast(self.device.type, enabled=use_cuda) if use_cuda else dummy_context():
            output = self.network(data)
            l = self._combine_loss(output, target, data)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }
