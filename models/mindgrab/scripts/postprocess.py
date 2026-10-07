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

"""Port of brainchopC's mindgrab post-segmentation chain (brainchopc.c's
`run_model`, the `mindgrab_output` branch, lines ~430-471): largest-component
cleanup, reslice back to the input's native grid, and the mask semantics --
mindgrab's output is the *original* input image with non-brain voxels
floored to the input's own minimum intensity, not a separate label volume.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from .conform import reslice_to_grid

__all__ = ["mindgrab_postprocess"]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """bwlabel.c `bwlabel_largest`, `binary_output=True` (mindgrab is a mask
    model, BC_MINDGRAB_OUTPUT_IS_MASK=1): 26-connected components, keep only
    the largest, binarize it to 1.0.
    """
    structure = np.ones((3, 3, 3), dtype=np.int8)
    labeled, num_labels = label(mask > 0.0, structure=structure)
    if num_labels == 0:
        return np.zeros_like(mask, dtype=np.float32)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    best_label = int(np.argmax(counts))
    return (labeled == best_label).astype(np.float32)


def mindgrab_postprocess(
    class_logits: np.ndarray, conform_affine: np.ndarray, orig_img: np.ndarray, orig_affine: np.ndarray
) -> np.ndarray:
    """
    Args:
        class_logits: network output, shape (2, 256, 256, 256) in conform
            space (channel 0 = background, channel 1 = brain).
        conform_affine: the 4x4 affine returned by `mindgrab_preprocess`.
        orig_img: the original input volume, shape (nx, ny, nz), untouched
            (not the intensity-scaled copy used for conforming).
        orig_affine: the original input volume's 4x4 voxel-to-world affine.

    Returns:
        float32 array shaped like `orig_img`: `orig_img` with every voxel
        outside the predicted brain mask set to `orig_img`'s own minimum
        value -- this is mindgrab's actual output, not a separate mask.
    """
    # meshnet_cpu.c `mn_classify`: argmax over classes, ties keep the lower
    # (background) index -- numpy's argmax already returns the first max.
    pred = np.argmax(class_logits, axis=0).astype(np.float32)
    pred = _largest_component(pred)

    # brainchopc_reslice(conformed, original, linear=0): nearest-neighbor,
    # source min (0.0, since pred is a 0/1 mask with background present) is
    # the out-of-FOV fill, matching `do_reslice`'s prefill.
    resliced_mask = reslice_to_grid(
        pred, conform_affine, np.asarray(orig_affine, dtype=np.float64), orig_img.shape, linear=False
    )

    out = np.asarray(orig_img, dtype=np.float32).copy()
    background_value = float(out.min())
    out[resliced_mask <= 0.0] = background_value
    return out
