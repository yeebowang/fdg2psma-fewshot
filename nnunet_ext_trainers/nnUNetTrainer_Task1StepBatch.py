"""
Task1 **stepbatch**：总 epoch 内全局 batch **线性**从 ``TASK1_STEPBATCH_BS_START`` 均匀降到
``TASK1_STEPBATCH_BS_END``（默认 18→6），多卡时沿用 nnU-Net 与单阶段相同的「余数分给前几张卡」规则
（例如全局 17 → 6+6+5）。

继承 ``nnUNetTrainer_Task1StdTrainVal50``（train/val 步数、段 checkpoint、跳过全量 val 等与其一致）。

环境变量（常用）:
  TASK1_NUM_EPOCHS           总轮数，默认 1000
  TASK1_STEPBATCH_BS_START   起始全局 batch，默认 18
  TASK1_STEPBATCH_BS_END     结束全局 batch，默认 6（须 >= world_size，3 卡时勿 <3）
  TASK1_TRAIN_NUM_GPUS       与 nnUNet ``-num_gpus`` 一致（默认 3）

说明：每 epoch 若全局 batch 与上一 epoch 不同，会结束旧 DataLoader 并重建（与 nnUNet ``on_train_start`` 同类逻辑）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.distributed as dist

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)

from nnunet_ext_trainers.nnUNetTrainer_Task1StdTrainVal50 import (
    nnUNetTrainer_Task1StdTrainVal50,
)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return int(str(v).strip())


class nnUNetTrainer_Task1StepBatch(nnUNetTrainer_Task1StdTrainVal50):
    """全局 batch 随 epoch 线性下降；多卡 per-GPU batch 与 nnUNetTrainer 一致（可不等分）。"""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        self._task1_bs_start = max(1, _env_int("TASK1_STEPBATCH_BS_START", 18))
        self._task1_bs_end = max(1, _env_int("TASK1_STEPBATCH_BS_END", 6))
        if self._task1_bs_end > self._task1_bs_start:
            self._task1_bs_start, self._task1_bs_end = self._task1_bs_end, self._task1_bs_start
        self._task1_last_built_global_bs: int | None = None
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )

    def _task1_global_batch_for_epoch(self, epoch: int) -> int:
        ne = max(1, int(self.num_epochs))
        e = max(0, min(int(epoch), ne - 1))
        if ne <= 1:
            return int(self._task1_bs_start)
        t = e / float(ne - 1)
        b = int(
            round(
                float(self._task1_bs_start)
                + (float(self._task1_bs_end) - float(self._task1_bs_start)) * t
            )
        )
        return int(max(self._task1_bs_end, min(self._task1_bs_start, b)))

    def _set_batch_size_and_oversample(self):
        """与 nnUNetTrainer 相同逻辑，但全局 batch 取自 ``_task1_global_batch_for_epoch(self.current_epoch)``。"""
        global_batch_size = self._task1_global_batch_for_epoch(int(self.current_epoch))
        if not self.is_ddp:
            self.batch_size = global_batch_size
            return

        world_size = dist.get_world_size()
        my_rank = dist.get_rank()
        assert global_batch_size >= world_size, (
            f"Task1StepBatch: global_batch_size={global_batch_size} < world_size={world_size}。"
            " 请提高 TASK1_STEPBATCH_BS_END 或减少 GPU 数。"
        )

        batch_size_per_GPU = [global_batch_size // world_size] * world_size
        batch_size_per_GPU = [
            batch_size_per_GPU[i] + 1
            if (batch_size_per_GPU[i] * world_size + i) < global_batch_size
            else batch_size_per_GPU[i]
            for i in range(len(batch_size_per_GPU))
        ]
        assert sum(batch_size_per_GPU) == global_batch_size

        sample_id_low = 0 if my_rank == 0 else int(np.sum(batch_size_per_GPU[:my_rank]))
        sample_id_high = int(np.sum(batch_size_per_GPU[: my_rank + 1]))

        oversample = [
            True
            if not i < round(global_batch_size * (1 - self.oversample_foreground_percent))
            else False
            for i in range(global_batch_size)
        ]

        if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
            oversample_percent = 0.0
        elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
            oversample_percent = 1.0
        else:
            oversample_percent = sum(oversample[sample_id_low:sample_id_high]) / float(
                batch_size_per_GPU[my_rank]
            )

        self.print_to_log_file(
            f"worker {my_rank} oversample {oversample_percent}",
            also_print_to_console=False,
        )
        self.print_to_log_file(
            f"worker {my_rank} batch_size {batch_size_per_GPU[my_rank]} "
            f"(global={global_batch_size}, split={batch_size_per_GPU})",
            also_print_to_console=False,
        )

        self.batch_size = batch_size_per_GPU[my_rank]
        self.oversample_foreground_percent = oversample_percent

    def _task1_finish_dataloaders(self) -> None:
        if self.dataloader_train is None and self.dataloader_val is None:
            return
        old_stdout = sys.stdout
        with open(os.devnull, "w") as f:
            sys.stdout = f
            try:
                if self.dataloader_train is not None and isinstance(
                    self.dataloader_train,
                    (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter),
                ):
                    self.dataloader_train._finish()
                if self.dataloader_val is not None and isinstance(
                    self.dataloader_val,
                    (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter),
                ):
                    self.dataloader_val._finish()
            finally:
                sys.stdout = old_stdout
        self.dataloader_train = self.dataloader_val = None

    def on_train_start(self):
        """checkpoint 恢复后 ``current_epoch`` 已就绪，在此按当前 epoch 重建 batch 再建 DataLoader。"""
        self._task1_finish_dataloaders()
        self._set_batch_size_and_oversample()
        self._task1_last_built_global_bs = self._task1_global_batch_for_epoch(
            int(self.current_epoch)
        )
        if self.local_rank == 0:
            self.print_to_log_file(
                f"[Task1StepBatch] ep={self.current_epoch} "
                f"global_bs={self._task1_last_built_global_bs} "
                f"(schedule {self._task1_bs_start}->{self._task1_bs_end} over {self.num_epochs} ep)"
            )
        super().on_train_start()

    def on_train_epoch_start(self):
        if int(self.current_epoch) > 0:
            gb = self._task1_global_batch_for_epoch(int(self.current_epoch))
            if gb != self._task1_last_built_global_bs:
                self._task1_finish_dataloaders()
                self._set_batch_size_and_oversample()
                self.dataloader_train, self.dataloader_val = self.get_dataloaders()
                _ = next(self.dataloader_train)
                _ = next(self.dataloader_val)
                self._task1_last_built_global_bs = gb
                if self.local_rank == 0:
                    self.print_to_log_file(
                        f"[Task1StepBatch] epoch {self.current_epoch}: "
                        f"rebuilt dataloaders global_bs={gb} (per-rank batch_size={self.batch_size})"
                    )
        super().on_train_epoch_start()
