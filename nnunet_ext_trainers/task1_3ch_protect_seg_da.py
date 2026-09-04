"""训练增强：暂存/恢复指定 image 通道，避免强度类 DA 改动二值 cascade seg（ch2）。"""
from __future__ import annotations

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class SaveImageChannelsTransform(BasicTransform):
    def __init__(self, channel_indices: tuple[int, ...], save_key: str = "task1_saved_image_channels"):
        super().__init__()
        self.channel_indices = tuple(channel_indices)
        self.save_key = save_key

    def apply(self, data_dict: dict, **params) -> dict:
        img = data_dict.get("image")
        if img is None:
            return data_dict
        n = int(img.shape[0])
        valid = tuple(c for c in self.channel_indices if 0 <= c < n)
        if not valid:
            return data_dict
        data_dict[self.save_key] = (valid, img[list(valid)].clone())
        return data_dict


class RestoreImageChannelsTransform(BasicTransform):
    def __init__(self, channel_indices: tuple[int, ...], save_key: str = "task1_saved_image_channels"):
        super().__init__()
        self.channel_indices = tuple(channel_indices)
        self.save_key = save_key

    def apply(self, data_dict: dict, **params) -> dict:
        img = data_dict.get("image")
        saved = data_dict.get(self.save_key)
        if img is None or saved is None:
            return data_dict
        if isinstance(saved, tuple) and len(saved) == 2:
            valid, tensors = saved
        else:
            valid = self.channel_indices
            tensors = saved
        for i, ch in enumerate(valid):
            if 0 <= ch < img.shape[0]:
                img[ch] = tensors[i]
        data_dict["image"] = img
        return data_dict
