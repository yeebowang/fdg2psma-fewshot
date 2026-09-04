"""Weighted MAE for binary segmentation (foreground probability vs GT)."""

from __future__ import annotations

from typing import Callable

import torch
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class WeightedMAELoss(nn.Module):
    """
    L = (1/|X|) * sum_v [ w_bg * 1(y=0) + w_fg * 1(y>0) ] * |p_fg(v) - y(v)|

    p_fg: softmax foreground probability; y: binary GT (0 background, 1 foreground).
    """

    def __init__(
        self,
        w_bg: float = 1.0,
        w_fg: float = 10.0,
        apply_nonlin: Callable | None = softmax_helper_dim1,
        ignore_label: int | None = None,
    ):
        super().__init__()
        self.w_bg = float(w_bg)
        self.w_fg = float(w_fg)
        self.apply_nonlin = apply_nonlin
        self.ignore_label = ignore_label

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.apply_nonlin is not None:
            prob = self.apply_nonlin(net_output)
        else:
            prob = net_output
        if prob.shape[1] < 2:
            raise ValueError(f"WeightedMAELoss expects >=2 classes, got {prob.shape[1]}")
        pred_fg = prob[:, 1]

        if target.ndim == pred_fg.ndim + 1:
            gt = target[:, 0]
        else:
            gt = target
        gt_bin = (gt > 0).float()

        if self.ignore_label is not None:
            valid = gt != self.ignore_label
            gt_bin = (gt > 0).float()
            w = torch.where(gt_bin > 0, self.w_fg, self.w_bg)
            abs_err = torch.abs(pred_fg - gt_bin) * valid
            w = w * valid
            denom = valid.sum().clamp_min(1e-8)
        else:
            w = torch.where(gt_bin > 0, self.w_fg, self.w_bg)
            abs_err = torch.abs(pred_fg - gt_bin)
            denom = torch.tensor(
                float(pred_fg.numel()), device=pred_fg.device, dtype=pred_fg.dtype
            )

        return (w * abs_err).sum() / denom
