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

__all__ = ["MindGrabNet"]

# Five passes over the same dilation cycle, the MeshNet dilation schedule
# this architecture was trained with.
DEFAULT_DILATIONS: Sequence[int] = (16, 8, 4, 2, 1) * 5


class MindGrabNet(nn.Module):
    """Dilated 3D conv stack used by the MindGrab skull-stripping model.

    A MeshNet-style network: a stack of same-padded, bias-free dilated
    Conv3d blocks, each followed by an affine-free instance normalization
    (per-channel mean/var computed live over the full spatial volume every
    forward pass, not a fixed running statistic -- brainchopC calls this
    "GroupNorm" because it is GroupNorm with num_groups == num_channels, eps
    1e-5, no learned scale/bias) and then GELU, followed by a 1x1x1
    classifier conv.

    The classifier conv is bias-free by construction: the upstream
    checkpoint carries a classifier bias, but the reference runtime never
    applies it, so it is not part of the computational graph and is not
    converted into this bundle's weights.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 15,
        out_channels: int = 2,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        c_in = in_channels
        for dilation in dilations:
            layers.append(nn.Conv3d(c_in, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False))
            layers.append(nn.InstanceNorm3d(channels, eps=1e-5, affine=False, track_running_stats=False))
            layers.append(nn.GELU(approximate="tanh"))
            c_in = channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Conv3d(channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
