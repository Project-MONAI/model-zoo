import math
import typing as t
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from cucim import CuImage
from huggingface_hub import PyTorchModelHubMixin
from torchvision.transforms import functional as TF
from torchvision.transforms import v2 as T

from networks.vit import vit4k_base, vit_base, vit_global_base
from utils.tensor_utils import (
    format_first_stg_act_as_second_stg_inp,
    format_second_stg_act_as_third_stg_inp,
    forward_with_batch_size_limit,
    scale_and_normalize,
    tile,
)
from utils.wsi_utils import load_slide_img, segment_tissue

if t.TYPE_CHECKING:
    from _typeshed import StrPath


class Transform(T.Transform):
    # For compatibility with torchvision <= 0.20
    def _transform(self, inpt, params):
        return self.transform(inpt, params)


class PadToDivisible(Transform):
    def __init__(self, size: int, pad_value: float | None = None):
        super().__init__()
        self.size = size
        self.pad_value = pad_value

    def transform(self, inpt, params):
        assert isinstance(inpt, torch.Tensor) and inpt.ndim >= 3

        H, W = inpt.shape[-2:]

        pad_h = (self.size - H % self.size) % self.size
        pad_w = (self.size - W % self.size) % self.size

        if pad_h > 0 or pad_w > 0:
            inpt = torch.nn.functional.pad(
                inpt, (0, pad_w, 0, pad_h), value=self.pad_value
            )

        return inpt


class EXAONEPathV20(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        small_tile_size: int = 256,
        large_tile_size: int = 4096,
    ):
        super().__init__()

        self.small_tile_size = small_tile_size
        self.large_tile_size = large_tile_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_first_stg = vit_base().to(self.device).eval()
        self.model_second_stg = vit4k_base().to(self.device).eval()
        self.model_third_stg = vit_global_base().to(self.device).eval()

    def forward(
        self,
        svs_path: "StrPath",
        target_mpp: float = 0.5,
        first_stg_batch_size: int = 128,
    ):
        small_tiles, is_tile_valid, padded_size, small_tile_size, large_tile_size = (
            self._load_wsi(svs_path, target_mpp=target_mpp)
        )
        width, height = padded_size

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                act1 = forward_with_batch_size_limit(
                    self.model_first_stg,
                    small_tiles,
                    batch_size_on_gpu=first_stg_batch_size,
                    preproc_fn=partial(
                        _preproc,
                        small_tile_size_with_this_mpp=small_tile_size,
                        small_tile_size_with_target_mpp=self.small_tile_size,
                    ),
                    device=self.device,
                    out_device="cpu",
                    dtype=torch.bfloat16,
                )
                act1 = act1.to(self.device)
                act1_formatted = format_first_stg_act_as_second_stg_inp(
                    act1,
                    height=height,
                    width=width,
                    small_tile_size=small_tile_size,
                    large_tile_size=large_tile_size,
                )
                act2: torch.Tensor = self.model_second_stg(act1_formatted)
                act2_formatted = format_second_stg_act_as_third_stg_inp(
                    act2,
                    height=height,
                    width=width,
                    large_tile_size=large_tile_size,
                )
                act3: torch.Tensor = self.model_third_stg(act2_formatted)

        return act1[is_tile_valid], act2, act3

    def _load_wsi(self, svs_path: "StrPath", target_mpp: float):
        svs_path = str(svs_path)

        # Load WSI tile
        with CuImage(str(svs_path)) as wsi_obj:
            try:
                mpp = float(wsi_obj.metadata["aperio"]["MPP"])
            except KeyError:
                print(
                    f"Warning: MPP metadata not found, using default value of {target_mpp}"
                )
                mpp = target_mpp

            img = load_slide_img(wsi_obj)
            height, width = img.shape[:2]
            mask_tensor = torch.from_numpy(
                segment_tissue(Path(svs_path), seg_level=-1)[0]
            )
            mask_tensor = TF.resize(mask_tensor.unsqueeze(0), [height, width]).squeeze(
                0
            )
            x: torch.Tensor = torch.from_numpy(img).permute(2, 0, 1)

        small_tile_size = math.ceil(self.small_tile_size * (target_mpp / mpp))
        large_tile_size = (
            self.large_tile_size // self.small_tile_size
        ) * small_tile_size
        pad_image = PadToDivisible(large_tile_size, 255)
        pad_mask = PadToDivisible(large_tile_size, 0)

        x = pad_image(x)
        padded_size = (x.size(-1), x.size(-2))

        x = tile(x, small_tile_size)
        mask_padded = pad_mask(mask_tensor.unsqueeze(0))
        mask_tile = tile(mask_padded, small_tile_size).squeeze(1)
        is_tile_valid = mask_tile.sum(dim=(1, 2)) > 0

        return x, is_tile_valid, padded_size, small_tile_size, large_tile_size


def _preproc(
    x: torch.Tensor,
    small_tile_size_with_this_mpp: int,
    small_tile_size_with_target_mpp: int,
):
    # Scale the input tensor to the target MPP
    if small_tile_size_with_this_mpp != small_tile_size_with_target_mpp:
        x = TF.resize(
            x,
            [small_tile_size_with_target_mpp, small_tile_size_with_target_mpp],
        )

    # Normalize the input tensor
    x = scale_and_normalize(x)

    return x
