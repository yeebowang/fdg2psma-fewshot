"""
External nnU-Net v2 trainer for fast train+val smoke tests (val dataloader / worker issues).

Install: set env nnUNet_extTrainer to this directory, then:
  nnUNetv2_train ... -tr nnUNetTrainer_Task1DemoSteps

See task1_train_nnunet_from_dataset1.sh (TASK1_DEMO_VAL_PROBE=1).

Full-volume sliding-window validation (``perform_actual_validation``) is skipped by default:
demo only needs the in-epoch train/val loops. Set ``TASK1_DEMO_ALLOW_FULL_VAL=1`` to run it
(e.g. debugging a fork that relies on exported val masks).
"""

from __future__ import annotations

import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _env_truthy(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class nnUNetTrainer_Task1DemoSteps(nnUNetTrainer):
    """Per-epoch train/val steps from env (default 1+1); ``num_epochs`` default 10 via ``TASK1_DEMO_NUM_EPOCHS``."""

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
        self.num_iterations_per_epoch = max(1, int(os.environ.get("TASK1_DEMO_TRAIN_STEPS", "1")))
        self.num_val_iterations_per_epoch = max(1, int(os.environ.get("TASK1_DEMO_VAL_STEPS", "1")))
        self.num_epochs = max(1, int(os.environ.get("TASK1_DEMO_NUM_EPOCHS", "10")))
        self.save_every = max(1, int(os.environ.get("TASK1_DEMO_SAVE_EVERY", "1")))
        if self.local_rank == 0:
            self.print_to_log_file(
                "[Task1DemoSteps] "
                f"num_iterations_per_epoch={self.num_iterations_per_epoch}, "
                f"num_val_iterations_per_epoch={self.num_val_iterations_per_epoch}, "
                f"num_epochs={self.num_epochs}, save_every={self.save_every}"
            )

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Skip full-dataset sliding-window val; demo uses only ``num_val_iterations_per_epoch``."""
        if not _env_truthy("TASK1_DEMO_ALLOW_FULL_VAL", default="0"):
            if self.local_rank == 0:
                self.print_to_log_file(
                    "[Task1DemoSteps] Skipping perform_actual_validation (full-volume val). "
                    "Set TASK1_DEMO_ALLOW_FULL_VAL=1 to enable."
                )
            return
        return super().perform_actual_validation(save_probabilities)
