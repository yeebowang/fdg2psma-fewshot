"""训练时从 4ch b2nd + 旁路 cascade fullres 二值 seg b2nd 拼成 5ch（mmap patch，对齐 cascade fullres）。"""
from __future__ import annotations

import os

import blosc2
from batchgenerators.utilities.file_and_folder_operations import isfile, join

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2


def task1_5ch_cascade_seg_b2nd_dir(preprocessed_folder: str) -> str:
    custom = os.environ.get("TASK1_5CH_CASCADE_SEG_B2ND_DIR", "").strip()
    if custom:
        return custom
    legacy = os.environ.get("TASK1_5CH_PROB_B2ND_DIR", "").strip()
    if legacy:
        return legacy
    return join(preprocessed_folder, "cascade_fullres_seg_uint8")


# 兼容旧 import 名
def task1_5ch_prob_b2nd_dir(preprocessed_folder: str) -> str:
    return task1_5ch_cascade_seg_b2nd_dir(preprocessed_folder)


class nnUNetDatasetBlosc2_Task1_5chSplitCascadeSeg(nnUNetDatasetBlosc2):
    """
    主 b2nd 为 CT+PET+FG+BG（4ch）；ch4 从 cascade_fullres_seg_uint8/{case}.b2nd uint8 读取。

    返回 mmap 二值标签图作第三返回值（同 nnUNet lowres seg_prev），由 dataloader patch crop
    后经 MoveSegAsOneHotToDataTransform 并入 image。
    """

    def __init__(self, folder: str, identifiers=None, folder_with_segs_from_previous_stage=None):
        self.cascade_seg_b2nd_folder = task1_5ch_cascade_seg_b2nd_dir(folder)
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage=None)

    def load_case(self, identifier):
        data, seg, _, properties = super().load_case(identifier)
        if data.shape[0] > 4:
            data = data[:4]
        elif data.shape[0] < 4:
            raise ValueError(
                f"expected >=4 image channels in {identifier}.b2nd, got {data.shape[0]}"
            )

        seg_path = join(self.cascade_seg_b2nd_folder, identifier + ".b2nd")
        if not isfile(seg_path):
            raise FileNotFoundError(f"missing cascade fullres seg b2nd: {seg_path}")

        dparams = {"nthreads": 1}
        cas_seg = blosc2.open(urlpath=seg_path, mode="r", dparams=dparams, mmap_mode="r")
        if cas_seg.ndim == 4 and cas_seg.shape[0] == 1:
            cas_seg = cas_seg[0]
        if cas_seg.ndim != 3:
            raise ValueError(f"bad cascade seg shape {cas_seg.shape} for {identifier}")

        if tuple(cas_seg.shape) != tuple(data.shape[1:]):
            raise ValueError(
                f"cascade seg spatial {cas_seg.shape} != image {data.shape[1:]} for {identifier}"
            )

        return data, seg, cas_seg, properties


# 兼容旧 trainer import
nnUNetDatasetBlosc2_Task1_5chSplitProb = nnUNetDatasetBlosc2_Task1_5chSplitCascadeSeg
