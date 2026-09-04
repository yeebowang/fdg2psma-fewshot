"""
[3.5/5] 五通道 fullres + Weighted MAE 分割 loss（禁用 CE / Dice）。

L = (1/|X|) sum [ w_bg*1(y=0) + w_fg*1(y>0) ] |p_fg - y|

环境变量:
  TASK1_WMAE_W_BG   默认 1
  TASK1_WMAE_W_FG   默认 10
  其余同 nnUNetTrainer_Task1StdTrainVal50_5chCascadeProb
"""

from __future__ import annotations

import os

import numpy as np
import torch

from nnunet_ext_trainers.task1_weighted_mae_loss import WeightedMAELoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunet_ext_trainers.nnUNetTrainer_Task1StdTrainVal50_5chCascadeProb import (
    nnUNetTrainer_Task1StdTrainVal50_5chCascadeProb,
)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(str(raw).strip())


class nnUNetTrainer_Task1StdTrainVal50_5chCascadeWMAE(nnUNetTrainer_Task1StdTrainVal50_5chCascadeProb):
    """5ch fullres；纯 W-MAE，无 CE/Dice/ref loss。"""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        self._wmae_w_bg = _env_float("TASK1_WMAE_W_BG", 1.0)
        self._wmae_w_fg = _env_float("TASK1_WMAE_W_FG", 10.0)
        if self.local_rank == 0:
            self.print_to_log_file(
                "[Task1StdTrainVal50_5chCascadeWMAE] "
                f"loss=W-MAE only (no CE/Dice) w_bg={self._wmae_w_bg} w_fg={self._wmae_w_fg} "
                f"ref_loss=0"
            )

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError(
                "nnUNetTrainer_Task1StdTrainVal50_5chCascadeWMAE: regions label not supported"
            )
        loss = WeightedMAELoss(
            w_bg=self._wmae_w_bg,
            w_fg=self._wmae_w_fg,
            ignore_label=self.label_manager.ignore_label,
        )
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
