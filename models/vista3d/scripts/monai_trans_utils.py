# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
from collections.abc import Hashable, Mapping

import numpy as np
import torch
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.transforms import MapLabelValue
from monai.transforms.transform import MapTransform
from monai.utils import look_up_option, min_version, optional_import
from monai.utils.type_conversion import convert_data_type, convert_to_cupy, convert_to_dst_type

measure, has_measure = optional_import("skimage.measure", "0.14.2", min_version)
cp, has_cp = optional_import("cupy")


def get_largest_connected_component_point(img: NdarrayTensor, point_coords=None, point_labels=None) -> NdarrayTensor:
    """
    Gets the largest connected component mask of an image. img is before post process! And will include NaN values.
    Args:
        img: [1, B, H, W, D]
        point_coords [B, N, 3]
        point_labels [B, N]
    """
    outs = torch.zeros_like(img)
    for c in range(len(point_coords)):
        if not ((point_labels[c] == 3).any() or (point_labels[c] == 1).any()):
            continue
        coords = point_coords[c, point_labels[c] == 3].tolist() + point_coords[c, point_labels[c] == 1].tolist()
        not_nan_mask = ~torch.isnan(img[0, c])
        img_ = torch.nan_to_num(img[0, c] > 0, 0)
        img_, *_ = convert_data_type(img_, np.ndarray)
        label = measure.label
        features = label(img_, connectivity=3)
        pos_mask = torch.from_numpy(img_).to(img.device) > 0
        # if num features less than max desired, nothing to do.
        features = torch.from_numpy(features).to(img.device)
        # generate a map with all pos points
        idx = []
        for p in coords:
            idx.append(features[round(p[0]), round(p[1]), round(p[2])].item())
        idx = list(set(idx))
        for i in idx:
            if i == 0:
                continue
            outs[0, c] += features == i
        outs = outs > 0
        # find negative mean value
        fill_in = img[0, c][torch.logical_and(~outs[0, c], not_nan_mask)].mean()
        img[0, c][torch.logical_and(pos_mask, ~outs[0, c])] = fill_in
    return img


def get_largest_connected_component_mask(
    img_pos: NdarrayTensor,
    img_neg: NdarrayTensor,
    connectivity: int | None = None,
    num_components: int = 1,
    point_coords=None,
    point_labels=None,
    margins=3,
) -> NdarrayTensor:
    """
    Gets the largest connected component mask of an image.

    Args:
        img: Image to get largest connected component from. Shape is (spatial_dim1 [, spatial_dim2, ...])
        connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
            Accepted values are ranging from  1 to input.ndim. If ``None``, a full
            connectivity of ``input.ndim`` is used. for more details:
            https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label.
        num_components: The number of largest components to preserve.
    """
    # use skimage/cucim.skimage and np/cp depending on whether packages are
    # available and input is non-cpu torch.tensor
    cucim_skimage, has_cucim = optional_import("cucim.skimage")

    use_cp = has_cp and has_cucim and isinstance(img_pos, torch.Tensor) and img_pos.device != torch.device("cpu")
    if use_cp:
        img_pos_ = convert_to_cupy(img_pos.short())  # type: ignore
        img_neg_ = convert_to_cupy(img_neg.short())  # type: ignore
        label = cucim_skimage.measure.label
        lib = cp
    else:
        if not has_measure:
            raise RuntimeError("Skimage.measure required.")
        img_pos_, *_ = convert_data_type(img_pos, np.ndarray)
        img_neg_, *_ = convert_data_type(img_neg, np.ndarray)
        label = measure.label
        lib = np

    # features will be an image -- 0 for background and then each different
    # feature will have its own index.
    # features, num_features = label(img_, connectivity=connectivity, return_num=True)

    features_pos, num_features = label(img_pos_, connectivity=3, return_num=True)
    features_neg, num_features = label(img_neg_, connectivity=3, return_num=True)

    # if num features less than max desired, nothing to do.
    outs = np.zeros_like(img_pos_)
    for bs in range(point_coords.shape[0]):
        for i, p in enumerate(point_coords[bs]):
            if point_labels[bs, i] == 1 or point_labels[bs, i] == 3:
                features = features_pos
            elif point_labels[bs, i] == 0 or point_labels[bs, i] == 2:
                features = features_neg
            else:
                # if -1 padding point, skip
                continue
            p = p.round().int()
            for margin in range(margins):
                l, r = max(p[0].item() - margin, 0), min(p[0].item() + margin + 1, features.shape[-3])
                t, d = max(p[1].item() - margin, 0), min(p[1].item() + margin + 1, features.shape[-2])
                f, b = max(p[2].item() - margin, 0), min(p[2].item() + margin + 1, features.shape[-1])
                index = features_pos[bs, 0, l:r, t:d, f:b].max()
                if index > 0:
                    outs[[bs]] += lib.isin(features[[bs]], index)
                    break
    outs[outs > 1] = 1
    outs = convert_to_dst_type(outs, dst=img_pos, dtype=outs.dtype)[0]
    return outs


class VistaPostTransform(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            dataset_transforms: a dictionary specifies the transform for corresponding dataset:
                key: dataset name, value: list of data transforms.
            dataset_key: key to get the dataset name from the data dictionary, default to "dataset_name".
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        for keys in self.keys:
            if keys in data:
                pred = data[keys]
                object_num = pred.shape[0]
                device = pred.device
                if data.get("label_prompt", None) is None and data.get("points", None) is not None:
                    pred = get_largest_connected_component_point(
                        pred.unsqueeze(0),
                        point_coords=data.get("points").to(device),
                        point_labels=data.get("point_labels").to(device),
                    )[0]
                pred[pred < 0] = 0.0
                # if it's multichannel, perform argmax
                if object_num > 1:
                    # concate background channel. Make sure user did not provide 0 as prompt.
                    is_bk = torch.all(pred <= 0, dim=0, keepdim=True)
                    pred = pred.argmax(0).unsqueeze(0).float() + 1.0
                    pred[is_bk] = 0.0
                else:
                    # AsDiscrete will remove NaN
                    # pred = monai.transforms.AsDiscrete(threshold=0.5)(pred)
                    pred[pred > 0] = 1.0
                if "label_prompt" in data and data["label_prompt"] is not None:
                    pred += 0.5  # inplace mapping to avoid cloning pred
                    for i in range(1, object_num + 1):
                        frac = i + 0.5
                        pred[pred == frac] = data["label_prompt"][i - 1].to(pred.dtype)
                    pred[pred == 0.5] = 0.0
                data[keys] = pred
        return data


def get_name_to_index_mapping(bundle_root):
    """get the label name to index mapping"""
    name_to_index_mapping = {}
    metadata_path = os.path.join(bundle_root, "configs/metadata.json")
    if not os.path.isfile(metadata_path):
        return name_to_index_mapping
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    labels = metadata.get("network_data_format", {}).get("outputs", {}).get("pred", {}).get("channel_def")
    if labels is None:
        return name_to_index_mapping
    name_to_index_mapping = {v.lower(): int(k) for k, v in labels.items()}
    return name_to_index_mapping


def convert_name_to_index(name_to_index_mapping, label_prompt):
    """convert the label name to index"""
    if label_prompt is not None and isinstance(label_prompt, list):
        converted_label_prompt = []
        # for new class, add to the mapping
        for l in label_prompt:
            if isinstance(l, str) and not l.isdigit():
                if l.lower() not in name_to_index_mapping:
                    name_to_index_mapping[l.lower()] = len(name_to_index_mapping)
        for l in label_prompt:
            if isinstance(l, (int, str)):
                converted_label_prompt.append(
                    name_to_index_mapping.get(l.lower(), int(l) if l.isdigit() else 0) if isinstance(l, str) else int(l)
                )
            else:
                converted_label_prompt.append(l)
        return converted_label_prompt
    return label_prompt


class VistaPreTransform(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        special_index=(25, 26, 27, 28, 29, 117),
        subclass=None,
        bundle_root=None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            dataset_transforms: a dictionary specifies the transform for corresponding dataset:
                key: dataset name, value: list of data transforms.
            dataset_key: key to get the dataset name from the data dictionary, default to "dataset_name".
            allow_missing_keys: don't raise exception if key is missing.
            special_index: the class index that need to be handled differently.
        """
        super().__init__(keys, allow_missing_keys)
        self.special_index = special_index
        self.subclass = subclass
        self.name_to_index_mapping = get_name_to_index_mapping(bundle_root)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        label_prompt = data.get("label_prompt", None)
        point_labels = data.get("point_labels", None)
        # convert the label name to index if needed
        label_prompt = convert_name_to_index(self.name_to_index_mapping, label_prompt)
        try:
            # The evaluator will check prompt. The invalid prompt will be skipped here and captured by evaluator.
            if self.subclass is not None and label_prompt is not None:
                _label_prompt = []
                subclass_keys = list(map(int, self.subclass.keys()))
                for i in range(len(label_prompt)):
                    if label_prompt[i] in subclass_keys:
                        _label_prompt.extend(self.subclass[str(label_prompt[i])])
                    else:
                        _label_prompt.append(label_prompt[i])
                data["label_prompt"] = _label_prompt

            if label_prompt is not None and point_labels is not None:
                if label_prompt[0] in self.special_index:
                    point_labels = np.array(point_labels)
                    point_labels[point_labels == 0] = 2
                    point_labels[point_labels == 1] = 3
                    point_labels = point_labels.tolist()
                data["point_labels"] = point_labels
        except Exception:
            pass

        return data


class RelabelD(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        label_mappings: dict[str, list[tuple[int, int]]],
        dtype: DtypeLike = np.int16,
        dataset_key: str = "dataset_name",
        allow_missing_keys: bool = False,
    ) -> None:
        """
        label_mappings[data[dataset_key]] should has the format: [(local label, global label), ...]

        This list of local -> global label mappings will be applied to each input `data[keys]`.
        if `data[dataset_key]` is not in `label_mappings`, label_mappings['default']` will be used.
        if `label_mappings[data[dataset_key]]` is None, no relabeling will be performed.

        Args:
            keys: keys of the corresponding items to be transformed.
            label_mappings: a dictionary specifies how local dataset class indices are mapped to the
                global class indices, format:
                key: dataset name, value: list of (local label, global label) pairs
                set `label_mappings={}` to completely skip this transform.
            dtype: convert the output data to dtype, default to float32.
            dataset_key: key to get the dataset name from the data dictionary, default to "dataset_name".
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.mappers = {}
        self.dataset_key = dataset_key
        for name, mapping in label_mappings.items():
            self.mappers[name] = MapLabelValue(
                orig_labels=[int(pair[0]) for pair in mapping],
                target_labels=[int(pair[1]) for pair in mapping],
                dtype=dtype,
            )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        dataset_name = d.get(self.dataset_key, "default")
        _m = look_up_option(dataset_name, self.mappers, default=None)
        if _m is None:
            return d
        for key in self.key_iterator(d):
            d[key] = _m(d[key])
        return d
