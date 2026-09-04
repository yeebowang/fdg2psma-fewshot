"""
Task1 正式训练：与 AutoPETV nnUNet baseline 相同的 epoch 总数与 val 步数，train 步数可由环境变量收紧。

PETV baseline（Dataset998 debug.json）：num_epochs=1000, num_iterations_per_epoch=250,
num_val_iterations_per_epoch=50。

本 Trainer 默认：1000 / 100 / 50，可通过环境变量覆盖。
"""

from __future__ import annotations

import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _env_int_nonempty(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return max(1, int(str(v).strip()))


class nnUNetTrainer_Task1PETVSchedule(nnUNetTrainer):
    """与 PETV baseline 对齐的 ep 与 val 步数；train 步数默认 100（非 250）。"""

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
        self.num_iterations_per_epoch = _env_int_nonempty(
            "TASK1_TRAIN_ITERS_PER_EPOCH", 100
        )
        self.num_val_iterations_per_epoch = _env_int_nonempty(
            "TASK1_VAL_ITERS_PER_EPOCH", 50
        )
        self.num_epochs = _env_int_nonempty("TASK1_NUM_EPOCHS", 1000)
        if self.local_rank == 0:
            self.print_to_log_file(
                "[Task1PETVSchedule] "
                f"num_iterations_per_epoch={self.num_iterations_per_epoch}, "
                f"num_val_iterations_per_epoch={self.num_val_iterations_per_epoch}, "
                f"num_epochs={self.num_epochs}"
            )
