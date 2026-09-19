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

"""Port of brainchopC's default (non---pve) post-segmentation chain for
label-output models like mindmap (brainchopc.c's `run_model`, the label-model
branch, lines ~565-624): largest-component cleanup and reslice back to the
input's native grid, producing an 18-class label map -- mindmap's output
*is* the label volume itself.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from .conform import reslice_to_grid

__all__ = ["mindmap_postprocess"]


def _largest_component_keep_labels(labels: np.ndarray) -> np.ndarray:
    """bwlabel.c `bwlabel_largest`, `binary_output=False` (mindmap is a
    labels model, BC_MINDMAP_OUTPUT_IS_MASK=0): a single 26-connected flood
    fill over *every* nonzero voxel regardless of its class value (i.e.
    connectivity does not distinguish between different label values), keep
    only the single largest such component, and zero every voxel outside it
    -- but voxels *inside* the retained component keep their original class
    label rather than being binarized.
    """
    structure = np.ones((3, 3, 3), dtype=np.int8)
    connected, num_components = label(labels > 0, structure=structure)
    if num_components == 0:
        return np.zeros_like(labels, dtype=np.float32)
    counts = np.bincount(connected.ravel())
    counts[0] = 0
    best_component = int(np.argmax(counts))
    out = labels.astype(np.float32).copy()
    out[connected != best_component] = 0.0
    return out


def mindmap_postprocess(
    class_logits: np.ndarray,
    conform_affine: np.ndarray,
    orig_shape: tuple[int, int, int],
    orig_affine: np.ndarray,
    save_conform: bool = False,
) -> np.ndarray:
    """
    Args:
        class_logits: network output, shape (18, 256, 256, 256) in conform
            space (channel 0 = background/"Unknown").
        conform_affine: the 4x4 affine returned by `mindmap_preprocess`.
        orig_shape: the original input volume's spatial shape (nx, ny, nz).
        orig_affine: the original input volume's 4x4 voxel-to-world affine.
        save_conform: if True, skip the reslice back to the input's native
            grid and return the label map in 256^3 conform space instead
            (brainchopC's `--save-conform`; mindmap's model_meta.json sets
            `save_conform: true` in its capabilities).

    Returns:
        float32 array of class labels (0-17): shaped like `orig_shape` by
        default, or (256, 256, 256) if `save_conform` is True.
    """
    # meshnet_cpu.c `mn_classify`: argmax over classes, ties keep the lower
    # (background) index -- numpy's argmax already returns the first max.
    pred = np.argmax(class_logits, axis=0).astype(np.float32)
    pred = _largest_component_keep_labels(pred)

    if save_conform:
        return pred

    # brainchopc_reslice(conformed, original, linear=0): nearest-neighbor,
    # source min (0.0, background) is the out-of-FOV fill, matching
    # `do_reslice`'s prefill.
    return reslice_to_grid(pred, conform_affine, np.asarray(orig_affine, dtype=np.float64), orig_shape, linear=False)
