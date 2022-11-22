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

import torch.nn as nn
from monai.networks.blocks import Warp
from monai.networks.nets import resnet18
from monai.networks.nets.regunet import AffineHead


class RegResNet(nn.Module):
    def __init__(
        self,
        image_size=(64, 64),
        spatial_dims=2,
        mod=None,
        mode="bilinear",
        padding_mode="border",
        features=400,  # feature dimension of `mod`
    ):
        super().__init__()
        self.features = resnet18(n_input_channels=2, spatial_dims=spatial_dims) if mod is None else mod
        self.affine_head = AffineHead(
            spatial_dims=spatial_dims, image_size=image_size, decode_size=[1] * spatial_dims, in_channels=features
        )
        self.warp = Warp(mode=mode, padding_mode=padding_mode)
        self.image_size = image_size

    def forward(self, x):
        self.features.to(device=x.device)
        self.affine_head.to(device=x.device)
        out = self.features(x)
        ddf = self.affine_head([out], self.image_size)
        f = self.warp(x[:, :1], ddf)  # warp the first channel
        return f
