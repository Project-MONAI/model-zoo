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

import torch
import torch.nn as nn
from monai.utils import optional_import
from PIL import Image
from torch.cuda.amp import autocast

from .utils import visualize_2d_image

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


class LDMSampler:
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def sampling_fn(
        self,
        input_noise: torch.Tensor,
        autoencoder_model: nn.Module,
        diffusion_model: nn.Module,
        scheduler: nn.Module,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)

        image = input_noise
        if conditioning is not None:
            cond_concat = conditioning.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            cond_concat = cond_concat.expand(list(cond_concat.shape[0:2]) + list(input_noise.shape[2:]))

        for t in progress_bar:
            with torch.no_grad():
                if conditioning is not None:
                    input_t = torch.cat((image, cond_concat), dim=1)
                else:
                    input_t = image
                model_output = diffusion_model(
                    input_t, timesteps=torch.Tensor((t,)).to(input_noise.device).long(), context=conditioning
                )
                image, _ = scheduler.step(model_output, t, image)

        with torch.no_grad():
            with autocast():
                sample = autoencoder_model.decode_stage_2_outputs(image)

        return sample

    def generate_image_np(self, sample) -> torch.Tensor:
        image_np = sample[0, 0].cpu().numpy().transpose(1, 0)[::-1, ::-1]
        return image_np

    def run(
        self,
        input_noise: torch.Tensor,
        autoencoder_model: nn.Module,
        diffusion_model: nn.Module,
        scheduler: nn.Module,
        output_file: str,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sample = self.sampling_fn(input_noise, autoencoder_model, diffusion_model, scheduler, conditioning)
        image_np = self.generate_image_np(sample)
        image_pil = Image.fromarray(visualize_2d_image(image_np), "RGB")
        image_pil.save(output_file)
