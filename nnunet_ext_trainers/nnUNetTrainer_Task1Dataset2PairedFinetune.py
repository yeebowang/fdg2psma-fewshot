"""
Task1 dataset2 finetune：train 集按 PSMA/FDG 配对采样——每个 batch 必为同一病人的 psma+fdg 成对出现。

环境变量（继承 Task1StdTrainVal50）:
  TASK1_DATASET2_PAIR_SUFFIX_PSMA  默认 _psma
  TASK1_DATASET2_PAIR_SUFFIX_FDG   默认 _fdg
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_Task1StdTrainVal50 import (
        nnUNetTrainer_Task1StdTrainVal50,
    )
except ImportError:
    from nnunet_ext_trainers.nnUNetTrainer_Task1StdTrainVal50 import (
        nnUNetTrainer_Task1StdTrainVal50,
    )


def _pair_map_from_keys(
    keys: list[str],
    suffix_psma: str,
    suffix_fdg: str,
) -> dict[str, tuple[str, str]]:
    psma_keys = {k for k in keys if k.endswith(suffix_psma)}
    fdg_keys = {k for k in keys if k.endswith(suffix_fdg)}
    pairs: dict[str, tuple[str, str]] = {}
    for pk in sorted(psma_keys):
        base = pk[: -len(suffix_psma)]
        fk = base + suffix_fdg
        if fk in fdg_keys:
            pairs[base] = (pk, fk)
    return pairs


class _PairAwareTrainDataLoader(nnUNetDataLoader):
    """batch_size 须为 2：每次返回同一 base 的 psma + fdg。"""

    def __init__(
        self,
        data: Any,
        batch_size: int,
        *args: Any,
        pair_map: dict[str, tuple[str, str]],
        **kwargs: Any,
    ):
        if batch_size != 2:
            raise ValueError(
                "PairAwareTrainDataLoader 要求 batch_size=2（1 对 tracer）"
            )
        super().__init__(data, batch_size, *args, **kwargs)
        if not pair_map:
            raise ValueError("pair_map 为空，无法配对采样")
        self._pair_ids = sorted(pair_map.keys())
        self._pair_map = pair_map

    def get_indices(self) -> list[str]:
        base = self._pair_ids[int(np.random.randint(len(self._pair_ids)))]
        psma, fdg = self._pair_map[base]
        return [psma, fdg]


class nnUNetTrainer_Task1Dataset2PairedFinetune(nnUNetTrainer_Task1StdTrainVal50):
    """dataset2 双 tracer 配对 finetune；val 仍走标准 dataloader。"""

    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._suffix_psma = os.environ.get(
            "TASK1_DATASET2_PAIR_SUFFIX_PSMA", "_psma"
        )
        self._suffix_fdg = os.environ.get("TASK1_DATASET2_PAIR_SUFFIX_FDG", "_fdg")
        self._pair_map: dict[str, tuple[str, str]] | None = None

    def get_dataloaders(self):
        from batchgenerators.dataloading.single_threaded_augmenter import (
            SingleThreadedAugmenter,
        )
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
            NonDetMultiThreadedAugmenter,
        )

        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

        _dl_tr, dl_val = super().get_dataloaders()
        dataset_tr, _dataset_val = self.get_tr_and_val_datasets()
        tr_keys = list(dataset_tr.identifiers)
        self._pair_map = _pair_map_from_keys(
            tr_keys, self._suffix_psma, self._suffix_fdg
        )
        if self.local_rank == 0:
            incomplete = len(tr_keys) - 2 * len(self._pair_map)
            self.print_to_log_file(
                f"[Task1Dataset2PairedFinetune] train keys={len(tr_keys)} "
                f"pairs={len(self._pair_map)} unpaired={incomplete}"
            )
            if incomplete:
                sample = [
                    k
                    for k in tr_keys
                    if not any(k in v for v in self._pair_map.values())
                ][:6]
                self.print_to_log_file(
                    f"[Task1Dataset2PairedFinetune] unpaired sample: {sample}"
                )

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions
            if self.label_manager.has_regions
            else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dl_tr = _PairAwareTrainDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            pair_map=self._pair_map,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            dl_tr = SingleThreadedAugmenter(dl_tr, None)
        else:
            dl_tr = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )

        _ = next(dl_tr)
        _ = next(dl_val)
        return dl_tr, dl_val
