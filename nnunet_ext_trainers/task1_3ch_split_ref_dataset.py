"""训练时从 2ch b2nd + 旁路 uint8 ref b2nd 拼成 3ch（对齐 cascade ch4 分离存储）。"""
from __future__ import annotations

import os

import blosc2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2


def task1_3ch_ref_b2nd_dir(preprocessed_folder: str) -> str:
    custom = os.environ.get("TASK1_3CH_REF_B2ND_DIR", "").strip()
    if custom:
        return custom
    return join(preprocessed_folder, "cascade_seg_ref_uint8")


class nnUNetDatasetBlosc2_Task1_3chSplitRef(nnUNetDatasetBlosc2):
    """主 b2nd 仅 CT+PET（2ch）；CASCADE_SEG 为 {ref_dir}/{case}.b2nd uint8，与 cascade ch4 同类。"""

    def __init__(self, folder: str, identifiers=None, folder_with_segs_from_previous_stage=None):
        self.ref_b2nd_folder = task1_3ch_ref_b2nd_dir(folder)
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage=None)

    def load_case(self, identifier):
        data, seg, _, properties = super().load_case(identifier)
        if data.shape[0] > 2:
            data = data[:2]
        elif data.shape[0] < 2:
            raise ValueError(
                f"expected >=2 image channels in {identifier}.b2nd, got {data.shape[0]}"
            )

        ref_path = join(self.ref_b2nd_folder, identifier + ".b2nd")
        if not isfile(ref_path):
            raise FileNotFoundError(f"missing cascade ref b2nd: {ref_path}")

        dparams = {"nthreads": 1}
        ref = blosc2.open(urlpath=ref_path, mode="r", dparams=dparams, mmap_mode="r")
        ref_np = np.asarray(ref)
        if ref_np.ndim == 4 and ref_np.shape[0] == 1:
            ref_np = ref_np[0]
        if ref_np.ndim != 3:
            raise ValueError(f"bad ref shape {ref_np.shape} for {identifier}")

        if tuple(ref_np.shape) != tuple(data.shape[1:]):
            raise ValueError(
                f"ref spatial {ref_np.shape} != image {data.shape[1:]} for {identifier}"
            )

        if ref_np.dtype == np.uint8:
            ref_f = ref_np.astype(np.float32, copy=False)
        else:
            ref_f = ref_np.astype(np.float32)
            if ref_f.max() > 1.0:
                thr = float(os.environ.get("TASK1_3CH_REF_BINARY_THRESHOLD", "0.5"))
                ref_f = (ref_f >= thr).astype(np.float32)

        data = np.vstack([np.asarray(data, dtype=np.float32), ref_f[np.newaxis]])
        return data, seg, None, properties
