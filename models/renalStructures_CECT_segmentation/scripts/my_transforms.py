import numpy as np
import torch
from monai.transforms import InvertibleTransform
from monai.transforms.transform import MapTransform


class ConcatImages(MapTransform, InvertibleTransform):
    def __init__(self, keys_merge, keys_out, allow_missing_keys=True):
        self.keys_merge = keys_merge
        self.keys_out = keys_out
        self.key_target_meta = keys_merge[0] + "_meta_dict"
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data):
        if isinstance(data, list):
            for data_row in data:
                data_row[self.keys_out] = np.concatenate([data_row[key] for key in self.keys_merge])
                data_row[self.keys_out + "_meta_dict"] = data_row[self.key_target_meta]
        else:
            data[self.keys_out] = np.concatenate([data[key] for key in self.keys_merge])
            data[self.keys_out + "_meta_dict"] = data[self.key_target_meta]
        return data

    def inverse(self, data):
        return data


class MergeClassesd(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            if key in data:
                num_classes = data[key].size(-4)
                device = data[key].device
                merged = None
                for channel in data[key].squeeze() * torch.tensor(list(range(num_classes)), device=device).view(
                    -1, 1, 1, 1
                ):
                    imgvol = channel
                    if merged is not None:
                        merged = merged + imgvol * ~((merged != 0) & (imgvol != 0))
                        data[key] = merged.unsqueeze(0)
                    else:
                        merged = imgvol
                        data[key] = imgvol.unsqueeze(0)
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"Key `{key}` of transform `{self.__class__.__name__}` was missing in the data"
                    " and allow_missing_keys==False."
                )
        return data
