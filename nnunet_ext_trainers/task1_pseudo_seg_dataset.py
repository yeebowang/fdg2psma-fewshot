"""训练时用旁路伪标签 ``{case}_seg.b2nd`` 替换 train GT（val 仍用原 GT）。"""
from __future__ import annotations

import os
from copy import deepcopy

import blosc2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2


def task1_pseudo_seg_b2nd_dir() -> str:
    return os.environ.get("TASK1_PSEUDO_SEG_B2ND_DIR", "").strip()


class nnUNetDatasetBlosc2_Task1PseudoSeg(nnUNetDatasetBlosc2):
    """
    从 ``TASK1_PSEUDO_SEG_B2ND_DIR/{id}_seg.b2nd`` 覆盖 seg，并可选加载
    ``{id}_classloc.npz`` 更新 ``properties['class_locations']``。
    """

    def __init__(self, folder: str, identifiers=None, folder_with_segs_from_previous_stage=None):
        self.pseudo_seg_folder = task1_pseudo_seg_b2nd_dir()
        if not self.pseudo_seg_folder:
            raise RuntimeError("TASK1_PSEUDO_SEG_B2ND_DIR is empty")
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)

    def load_case(self, identifier):
        data, _seg_unused, seg_prev, properties = super().load_case(identifier)
        seg_path = join(self.pseudo_seg_folder, identifier + "_seg.b2nd")
        if not isfile(seg_path):
            # 兼容仅写 {case}.b2nd
            alt = join(self.pseudo_seg_folder, identifier + ".b2nd")
            if isfile(alt):
                seg_path = alt
            else:
                raise FileNotFoundError(f"missing pseudo seg b2nd: {seg_path}")

        dparams = {"nthreads": 1}
        seg = blosc2.open(urlpath=seg_path, mode="r", dparams=dparams, mmap_mode="r")
        # 期望 (1,D,H,W)；若 (D,H,W) 则扩一维
        if getattr(seg, "ndim", 3) == 3:
            # mmap 不方便 expand；读成 numpy 再加通道（伪标通常稀疏，体积可接受）
            seg_np = np.asarray(seg)[None, ...]
            seg = seg_np

        props = deepcopy(properties) if isinstance(properties, dict) else properties
        cl_path = join(self.pseudo_seg_folder, identifier + "_classloc.npz")
        if isfile(cl_path):
            z = np.load(cl_path)
            cl = {}
            for k in z.files:
                lab = int(k[1:]) if k.startswith("c") and k[1:].isdigit() else k
                cl[lab] = np.asarray(z[k])
            if isinstance(props, dict):
                props["class_locations"] = cl
        else:
            # 即时重算（较慢，仅兜底）
            seg_np = np.asarray(seg)
            coords = np.argwhere(seg_np == 1)
            if coords.shape[0] > 10000:
                rs = np.random.RandomState(0)
                coords = coords[rs.choice(coords.shape[0], 10000, replace=False)]
            if isinstance(props, dict):
                props["class_locations"] = {
                    1: coords.astype(np.int64) if coords.size else np.zeros((0, 4), np.int64)
                }

        return data, seg, seg_prev, props
