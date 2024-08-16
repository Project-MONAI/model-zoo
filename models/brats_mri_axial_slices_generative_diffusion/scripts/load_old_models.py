# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import json
import os
import logging
import torch
import torch.distributed as dist
from monai.transforms import Compose, EnsureTyped, Lambdad, LoadImaged, Orientationd
from monai.data import DataLoader, Dataset, NibabelWriter, DistributedSampler
from monai.apps import download_url
from monai.data.utils import no_collation

def load_diffusion_ckpt(new_state_dict: dict, old_state_dict: dict, verbose=False) -> dict:
    """
    Load a state dict from a DiffusionModelUNet trained with
    [MONAI Generative](https://github.com/Project-MONAI/GenerativeModels).

    The loaded state dict is for
    MONAI DiffusionModelUNetMaisi and DiffusionModelUNet.

    Args:
        new_state_dict: state dict from the new model.
        old_state_dict: state dict from the old model.
    """
    if verbose:
        # print all new_state_dict keys that are not in old_state_dict
        for k in new_state_dict:
            if k not in old_state_dict:
                logging.info(f"New key {k} not found in old state dict")
        # and vice versa
        for k in old_state_dict:
            if k not in new_state_dict:
                logging.info(f"Old key {k} not found in new state dict")

    # copy over all matching keys
    for k in new_state_dict:
        if k in old_state_dict:
            new_state_dict[k] = old_state_dict.pop(k)

    # fix the attention blocks
    attention_blocks = [k.replace(".attn.to_k.weight", "") for k in new_state_dict if "attn.to_k.weight" in k]
    for block in attention_blocks:
        new_state_dict[f"{block}.attn.to_q.weight"] = old_state_dict.pop(f"{block}.to_q.weight")
        new_state_dict[f"{block}.attn.to_k.weight"] = old_state_dict.pop(f"{block}.to_k.weight")
        new_state_dict[f"{block}.attn.to_v.weight"] = old_state_dict.pop(f"{block}.to_v.weight")
        new_state_dict[f"{block}.attn.to_q.bias"] = old_state_dict.pop(f"{block}.to_q.bias")
        new_state_dict[f"{block}.attn.to_k.bias"] = old_state_dict.pop(f"{block}.to_k.bias")
        new_state_dict[f"{block}.attn.to_v.bias"] = old_state_dict.pop(f"{block}.to_v.bias")

        # projection
        new_state_dict[f"{block}.attn.out_proj.weight"] = old_state_dict.pop(f"{block}.proj_attn.weight")
        new_state_dict[f"{block}.attn.out_proj.bias"] = old_state_dict.pop(f"{block}.proj_attn.bias")

    # fix the upsample conv blocks which were renamed postconv
    for k in new_state_dict:
        if "postconv" in k:
            old_name = k.replace("postconv", "conv")
            new_state_dict[k] = old_state_dict.pop(old_name)

    if len(old_state_dict.keys()) > 0:
        logging.info(f"{old_state_dict.keys()} remaining***********")
    return new_state_dict
