import os
from typing import List

import numpy as np
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from nnunetv2.training.dataloading.utils import get_case_identifiers

try:
    import blosc2
except ImportError:  # pragma: no cover
    blosc2 = None


class nnUNetDataset(object):
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):
        """
        This does not actually load the dataset. It merely creates a dictionary where the keys are training case names and
        the values are dictionaries containing the relevant information for that case.

        Loading preference (aligned with modern nnU-Net / Dataset228):
          1) blosc2 ``.b2nd`` (chunked, patch-friendly)
          2) mmap ``.npy`` (legacy unpacked)
          3) ``.npz``
        """
        super().__init__()
        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()

        self.folder = folder
        self.dataset = {}
        for c in case_identifiers:
            self.dataset[c] = {}
            # keep .npz path for legacy fields; load_case prefers .b2nd when present
            self.dataset[c]['data_file'] = join(folder, f"{c}.npz")
            self.dataset[c]['properties_file'] = join(folder, f"{c}.pkl")
            self.dataset[c]['b2nd_file'] = join(folder, f"{c}.b2nd")
            self.dataset[c]['b2nd_seg_file'] = join(folder, f"{c}_seg.b2nd")
            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(
                    folder_with_segs_from_previous_stage, f"{c}.npz"
                )
                self.dataset[c]['seg_from_prev_stage_b2nd'] = join(
                    folder_with_segs_from_previous_stage, f"{c}.b2nd"
                )

        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for i in self.dataset.keys():
                self.dataset[i]['properties'] = load_pickle(self.dataset[i]['properties_file'])

        self.keep_files_open = ('nnUNet_keep_files_open' in os.environ.keys()) and \
                               (os.environ['nnUNet_keep_files_open'].lower() in ('true', '1', 't'))
        if blosc2 is not None:
            blosc2.set_nthreads(1)

    def __getitem__(self, key):
        ret = {**self.dataset[key]}
        if 'properties' not in ret.keys():
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret

    def __setitem__(self, key, value):
        return self.dataset.__setitem__(key, value)

    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return self.dataset.__len__()

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()

    @staticmethod
    def _open_b2nd(path: str):
        if blosc2 is None:
            raise ImportError(
                f"Found {path} but blosc2 is not installed; cannot use nnU-Net blosc2 pipeline"
            )
        return blosc2.open(urlpath=path, mode='r', dparams={'nthreads': 1}, mmap_mode='r')

    def load_case(self, key):
        entry = self[key]

        # 1) blosc2 (same path as Baseline1 / Dataset228)
        if isfile(entry['b2nd_file']) and isfile(entry['b2nd_seg_file']):
            if 'open_data_file' in entry.keys():
                data = entry['open_data_file']
            else:
                data = self._open_b2nd(entry['b2nd_file'])
                if self.keep_files_open:
                    self.dataset[key]['open_data_file'] = data
            if 'open_seg_file' in entry.keys():
                seg = entry['open_seg_file']
            else:
                seg = self._open_b2nd(entry['b2nd_seg_file'])
                if self.keep_files_open:
                    self.dataset[key]['open_seg_file'] = seg
        # 2) legacy unpacked npy
        elif isfile(entry['data_file'][:-4] + ".npy"):
            if 'open_data_file' in entry.keys():
                data = entry['open_data_file']
            else:
                data = np.load(entry['data_file'][:-4] + ".npy", 'r')
                if self.keep_files_open:
                    self.dataset[key]['open_data_file'] = data
            if 'open_seg_file' in entry.keys():
                seg = entry['open_seg_file']
            else:
                seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
                if self.keep_files_open:
                    self.dataset[key]['open_seg_file'] = seg
        # 3) npz
        else:
            data = np.load(entry['data_file'])['data']
            seg = np.load(entry['data_file'])['seg']

        if 'seg_from_prev_stage_file' in entry.keys():
            if isfile(entry.get('seg_from_prev_stage_b2nd', '')):
                seg_prev = self._open_b2nd(entry['seg_from_prev_stage_b2nd'])
            elif isfile(entry['seg_from_prev_stage_file'][:-4] + ".npy"):
                seg_prev = np.load(entry['seg_from_prev_stage_file'][:-4] + ".npy", 'r')
            else:
                seg_prev = np.load(entry['seg_from_prev_stage_file'])['seg']
            seg = np.vstack((np.asarray(seg), np.asarray(seg_prev)[None]))

        return data, seg, entry['properties']


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset003_Liver/3d_lowres'
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=0)
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
