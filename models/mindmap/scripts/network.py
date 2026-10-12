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

from collections.abc import Sequence

import torch
import torch.nn as nn

__all__ = ["MindMapNet"]

# The symmetric coprime dilation ramp the upstream reference implementation
# calls "gn_hdc_deep": receptive field 255 on a 256^3 grid (no gridding
# holes), shared verbatim with model16chan18cls -- mindmap is that same
# topology at 24 channels.
DEFAULT_DILATIONS: Sequence[int] = (1, 3, 5, 7, 13, 19, 31, 19, 13, 7, 5, 3, 1)


class MindMapNet(nn.Module):
    """Dilated 3D conv stack used by the MindMap 18-class brain segmentation
    model ("model24chan18cls" topology, 24 channels, published as
    model24chan18cls_gdice_prio in neuroneural/brainchop-test).

    A MeshNet-style network: 13 same-padded, bias-free dilated Conv3d blocks,
    each followed by GroupNorm (num_groups == num_channels, eps 1e-5, *with*
    learnable per-channel affine) and GELU, followed by a 1x1x1 classifier
    conv that does carry a bias.

    The upstream reference implementation's own weight layout represents
    each block's GroupNorm affine as a diagonal `[C, C]` 1x1x1 "affine"
    convolution (needed because its TFJS runtime has no rank-5
    BatchNorm/GroupNorm op), but that is just an alternate encoding of the
    same per-channel `gamma * x + beta`: see `scripts/checkpoint.py` for the
    reverse mapping back to `nn.GroupNorm`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 24,
        out_channels: int = 18,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        c_in = in_channels
        for dilation in dilations:
            layers.append(nn.Conv3d(c_in, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False))
            layers.append(nn.GroupNorm(num_groups=channels, num_channels=channels, eps=1e-5, affine=True))
            layers.append(nn.GELU(approximate="tanh"))
            c_in = channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Conv3d(channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
