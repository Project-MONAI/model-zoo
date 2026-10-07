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

"""MONAI MapTransform wrappers around conform.py/postprocess.py, so the
ported preprocessing and postprocessing can sit inside a bundle's
`configs/inference.json` transform Compose.

The extra state postprocessing needs (the original image's affine and
spatial shape, and the conform-space affine) is embedded in the preprocessed
image's own MetaTensor.meta dict rather than as sibling keys in the data
dict. This is deliberate: engines such as SupervisedEvaluator rebuild
`engine.state.output` as `{"image": inputs, "label": targets, "pred": ...}`
each iteration, discarding every other dict key -- but MetaTensor.meta
propagates through plain nn.Module forward passes and DataLoader collation,
so meta attached to "image" survives onto "pred" (network(inputs) keeps
inputs' meta) and is still there when postprocessing runs.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping

import numpy as np
import torch
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import MapTransform

from .conform import mindmap_preprocess
from .postprocess import mindmap_postprocess

__all__ = ["MindMapPreprocessd", "MindMapPostprocessd"]

_ORIG_SHAPE_KEY = "mindmap_orig_shape"
_ORIG_AFFINE_KEY = "mindmap_orig_affine"
_CONFORM_AFFINE_KEY = "mindmap_conform_affine"


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _drop_channel(arr: np.ndarray) -> np.ndarray:
    # LoadImaged(ensure_channel_first=True) yields (1, nx, ny, nz); conform.py
    # and postprocess.py work on plain (nx, ny, nz) volumes.
    return arr[0] if arr.ndim == 4 else arr


def _unbatch(arr: np.ndarray, ndim: int) -> np.ndarray:
    # DataLoader collation (batch_size=1) prepends a batch dim to anything it
    # stacks, including values stashed inside MetaTensor.meta.
    return arr[0] if arr.ndim == ndim + 1 else arr


class MindMapPreprocessd(MapTransform):
    """Runs mindmap_preprocess on each keyed image (a MetaTensor from
    LoadImaged). Replaces the image with a (1, 256, 256, 256) float32
    MetaTensor holding the conformed+normalized volume ready for
    MindMapNet, with the original image's shape, its affine, and the
    conform-space affine embedded in its `.meta` for MindMapPostprocessd to
    read back later in the pipeline (see module docstring for why).
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, object]) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            affine = _to_numpy(img.affine).astype(np.float64) if hasattr(img, "affine") else np.eye(4)
            arr = _drop_channel(_to_numpy(img).astype(np.float32))

            normalized, conform_affine = mindmap_preprocess(arr, affine)

            meta = dict(img.meta) if hasattr(img, "meta") else {}
            meta[_ORIG_SHAPE_KEY] = np.asarray(arr.shape)
            meta[_ORIG_AFFINE_KEY] = affine
            meta[_CONFORM_AFFINE_KEY] = conform_affine
            d[key] = MetaTensor(normalized[None], affine=torch.as_tensor(conform_affine), meta=meta)
        return d


class MindMapPostprocessd(MapTransform):
    """Runs mindmap_postprocess on each keyed network-output MetaTensor (18
    channels, conform space), using the original image's shape/affine
    MindMapPreprocessd embedded in its `.meta` (propagated here through the
    network forward pass and DataLoader collation). Replaces the entry with
    a (1, nx, ny, nz) MetaTensor in the input's native geometry (correct
    affine + meta attached, so a downstream SaveImaged writes a
    properly-headed, correctly-named output) holding the 18-class label map,
    unless `save_conform` is set, in which case the 256^3 conform-space
    labels are kept instead.
    """

    def __init__(self, keys: KeysCollection, save_conform: bool = False, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.save_conform = save_conform

    def __call__(self, data: Mapping[Hashable, object]) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            pred = d[key]
            meta = dict(pred.meta) if hasattr(pred, "meta") else {}
            if _ORIG_SHAPE_KEY not in meta:
                raise KeyError(
                    f"MindMapPostprocessd: '{key}' has no '{_ORIG_SHAPE_KEY}' meta entry -- "
                    "MindMapPreprocessd must run earlier in the pipeline on the same MetaTensor."
                )

            logits_np = _to_numpy(pred).astype(np.float32)
            logits_np = _unbatch(logits_np, ndim=4)  # (1, 18, 256, 256, 256) -> (18, 256, 256, 256)

            orig_shape = tuple(int(v) for v in _unbatch(_to_numpy(meta[_ORIG_SHAPE_KEY]), ndim=1))
            orig_affine = _to_numpy(meta[_ORIG_AFFINE_KEY]).astype(np.float64)
            conform_affine = _to_numpy(meta[_CONFORM_AFFINE_KEY]).astype(np.float64)
            orig_affine = _unbatch(orig_affine, ndim=2)
            conform_affine = _unbatch(conform_affine, ndim=2)

            out = mindmap_postprocess(
                logits_np, conform_affine, orig_shape, orig_affine, save_conform=self.save_conform
            )
            out_affine = conform_affine if self.save_conform else orig_affine

            out_meta = {
                k: v for k, v in meta.items() if k not in (_ORIG_SHAPE_KEY, _ORIG_AFFINE_KEY, _CONFORM_AFFINE_KEY)
            }
            out_meta["affine"] = torch.as_tensor(out_affine)
            d[key] = MetaTensor(out[None], affine=out_meta["affine"], meta=out_meta)
        return d
