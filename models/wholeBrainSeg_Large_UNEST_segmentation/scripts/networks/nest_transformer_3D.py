#!/usr/bin/env python3

# =========================================================================
# Adapted from https://github.com/google-research/nested-transformer.
# which has the following license...
# https://github.com/pytorch/vision/blob/main/LICENSE
#
# BSD 3-Clause License


# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Nested Transformer (NesT) in PyTorch
A PyTorch implement of Aggregating Nested Transformers as described in:
'Aggregating Nested Transformers'
    - https://arxiv.org/abs/2105.12723
The official Jax code is released and available at https://github.com/google-research/nested-transformer.
The weights have been converted with convert/convert_nest_flax.py
Acknowledgments:
* The paper authors for sharing their research, code, and model weights
* Ross Wightman's existing code off which I based this
Copyright 2021 Alexander Soare

"""

import collections.abc
import logging
import math
from functools import partial
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .nest import DropPath, Mlp, _assert, create_conv3d, create_pool3d, to_ntuple, trunc_normal_
from .patchEmbed3D import PatchEmbed3D

_logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """
    This is much like `.vision_transformer.Attention` but uses *localised* self attention by accepting an input with
     an extra "image block" dim
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x is shape: B (batch_size), T (image blocks), N (seq length per image block), C (embed dim)
        """
        b, t, n, c = x.shape
        # result of next line is (qkv, B, num (H)eads, T, N, (C')hannels per head)
        qkv = self.qkv(x).reshape(b, t, n, 3, self.num_heads, c // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 3, 4, 1).reshape(b, t, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x  # (B, T, N, C)


class TransformerLayer(nn.Module):
    """
    This is much like `.vision_transformer.Block` but:
        - Called TransformerLayer here to allow for "block" as defined in the paper ("non-overlapping image blocks")
        - Uses modified Attention layer that handles the "block" dimension
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.drop_path(self.attn(y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvPool(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, pad_type=""):
        super().__init__()
        self.conv = create_conv3d(in_channels, out_channels, kernel_size=3, padding=pad_type, bias=True)
        self.norm = norm_layer(out_channels)
        self.pool = create_pool3d("max", kernel_size=3, stride=2, padding=pad_type)

    def forward(self, x):
        """
        x is expected to have shape (B, C, D, H, W)
        """
        _assert(x.shape[-3] % 2 == 0, "BlockAggregation requires even input spatial dims")
        _assert(x.shape[-2] % 2 == 0, "BlockAggregation requires even input spatial dims")
        _assert(x.shape[-1] % 2 == 0, "BlockAggregation requires even input spatial dims")

        # print('In ConvPool x : {}'.format(x.shape))
        x = self.conv(x)
        # Layer norm done over channel dim only
        x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = self.pool(x)
        return x  # (B, C, D//2, H//2, W//2)


def blockify(x, block_size: int):
    """image to blocks
    Args:
        x (Tensor): with shape (B, D, H, W, C)
        block_size (int): edge length of a single square block in units of D, H, W
    """
    b, d, h, w, c = x.shape
    _assert(d % block_size == 0, "`block_size` must divide input depth evenly")
    _assert(h % block_size == 0, "`block_size` must divide input height evenly")
    _assert(w % block_size == 0, "`block_size` must divide input width evenly")
    grid_depth = d // block_size
    grid_height = h // block_size
    grid_width = w // block_size
    x = x.reshape(b, grid_depth, block_size, grid_height, block_size, grid_width, block_size, c)

    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
        b, grid_depth * grid_height * grid_width, -1, c
    )  # shape [2, 512, 27, 128]

    return x  # (B, T, N, C)


# @register_notrace_function  # reason: int receives Proxy
def deblockify(x, block_size: int):
    """blocks to image
    Args:
        x (Tensor): with shape (B, T, N, C) where T is number of blocks and N is sequence size per block
        block_size (int): edge length of a single square block in units of desired D, H, W
    """
    b, t, _, c = x.shape
    grid_size = round(math.pow(t, 1 / 3))
    depth = height = width = grid_size * block_size
    x = x.reshape(b, grid_size, grid_size, grid_size, block_size, block_size, block_size, c)

    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, depth, height, width, c)

    return x  # (B, D, H, W, C)


class NestLevel(nn.Module):
    """Single hierarchical level of a Nested Transformer"""

    def __init__(
        self,
        num_blocks,
        block_size,
        seq_length,
        num_heads,
        depth,
        embed_dim,
        prev_embed_dim=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rates: Sequence[int] = (),
        norm_layer=None,
        act_layer=None,
        pad_type="",
    ):
        super().__init__()
        self.block_size = block_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_blocks, seq_length, embed_dim))

        if prev_embed_dim is not None:
            self.pool = ConvPool(prev_embed_dim, embed_dim, norm_layer=norm_layer, pad_type=pad_type)
        else:
            self.pool = nn.Identity()

        # Transformer encoder
        if len(drop_path_rates):
            assert len(drop_path_rates) == depth, "Must provide as many drop path rates as there are transformer layers"
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerLayer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rates[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        """
        expects x as (B, C, D, H, W)
        """
        x = self.pool(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, H', W', C), switch to channels last for transformer

        x = blockify(x, self.block_size)  # (B, T, N, C')
        x = x + self.pos_embed

        x = self.transformer_encoder(x)  # (B, ,T, N, C')

        x = deblockify(x, self.block_size)  # (B, D', H', W', C') [2, 24, 24, 24, 128]
        # Channel-first for block aggregation, and generally to replicate convnet feature map at each stage
        return x.permute(0, 4, 1, 2, 3)  # (B, C, D', H', W')


class NestTransformer3D(nn.Module):
    """Nested Transformer (NesT)
    A PyTorch impl of : `Aggregating Nested Transformers`
        - https://arxiv.org/abs/2105.12723
    """

    def __init__(
        self,
        img_size=96,
        in_chans=1,
        patch_size=2,
        num_levels=3,
        embed_dims=(128, 256, 512),
        num_heads=(4, 8, 16),
        depths=(2, 2, 20),
        num_classes=1000,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.5,
        norm_layer=None,
        act_layer=None,
        pad_type="",
        weight_init="",
        global_pool="avg",
    ):
        """
        Args:
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            patch_size (int): patch size
            num_levels (int): number of block hierarchies (T_d in the paper)
            embed_dims (int, tuple): embedding dimensions of each level
            num_heads (int, tuple): number of attention heads for each level
            depths (int, tuple): number of transformer layers for each level
            num_classes (int): number of classes for classification head
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim for MLP of transformer layers
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate for MLP of transformer layers, MSA final projection layer, and classifier
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer for transformer layers
            act_layer: (nn.Module): activation layer in MLP of transformer layers
            pad_type: str: Type of padding to use '' for PyTorch symmetric, 'same' for TF SAME
            weight_init: (str): weight init scheme
            global_pool: (str): type of pooling operation to apply to final feature map
        Notes:
            - Default values follow NesT-B from the original Jax code.
            - `embed_dims`, `num_heads`, `depths` should be ints or tuples with length `num_levels`.
            - For those following the paper, Table A1 may have errors!
                - https://github.com/google-research/nested-transformer/issues/2
        """
        super().__init__()

        for param_name in ["embed_dims", "num_heads", "depths"]:
            param_value = locals()[param_name]
            if isinstance(param_value, collections.abc.Sequence):
                assert len(param_value) == num_levels, f"Require `len({param_name}) == num_levels`"

        embed_dims = to_ntuple(num_levels)(embed_dims)
        num_heads = to_ntuple(num_levels)(num_heads)
        depths = to_ntuple(num_levels)(depths)
        self.num_classes = num_classes
        self.num_features = embed_dims[-1]
        self.feature_info = []
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.drop_rate = drop_rate
        self.num_levels = num_levels
        if isinstance(img_size, collections.abc.Sequence):
            assert img_size[0] == img_size[1], "Model only handles square inputs"
            img_size = img_size[0]
        assert img_size % patch_size == 0, "`patch_size` must divide `img_size` evenly"
        self.patch_size = patch_size

        # Number of blocks at each level
        self.num_blocks = (8 ** torch.arange(num_levels)).flip(0).tolist()
        assert (img_size // patch_size) % round(
            math.pow(self.num_blocks[0], 1 / 3)
        ) == 0, "First level blocks don't fit evenly. Check `img_size`, `patch_size`, and `num_levels`"

        # Block edge size in units of patches
        # Hint: (img_size // patch_size) gives number of patches along edge of image. sqrt(self.num_blocks[0]) is the
        #  number of blocks along edge of image
        self.block_size = int((img_size // patch_size) // round(math.pow(self.num_blocks[0], 1 / 3)))

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=[img_size, img_size, img_size],
            patch_size=[patch_size, patch_size, patch_size],
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.num_patches = self.patch_embed.num_patches
        self.seq_length = self.num_patches // self.num_blocks[0]
        # Build up each hierarchical level
        levels = []

        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = None
        curr_stride = 4
        for i in range(len(self.num_blocks)):
            dim = embed_dims[i]
            levels.append(
                NestLevel(
                    self.num_blocks[i],
                    self.block_size,
                    self.seq_length,
                    num_heads[i],
                    depths[i],
                    dim,
                    prev_dim,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    dp_rates[i],
                    norm_layer,
                    act_layer,
                    pad_type=pad_type,
                )
            )
            self.feature_info += [dict(num_chs=dim, reduction=curr_stride, module=f"levels.{i}")]
            prev_dim = dim
            curr_stride *= 2

        self.levels = nn.ModuleList([levels[i] for i in range(num_levels)])

        # Final normalization layer
        self.norm = norm_layer(embed_dims[-1])

        self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode in ("nlhb", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        for level in self.levels:
            trunc_normal_(level.pos_embed, std=0.02, a=-2, b=2)
        named_apply(partial(_init_nest_weights, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {f"level.{i}.pos_embed" for i in range(len(self.levels))}

    def get_classifier(self):
        return self.head

    def forward_features(self, x):
        """x shape (B, C, D, H, W)"""
        x = self.patch_embed(x)

        hidden_states_out = [x]

        for _, level in enumerate(self.levels):
            x = level(x)
            hidden_states_out.append(x)
        # Layer norm done over channel dim only (to NDHWC and back)
        x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x, hidden_states_out

    def forward(self, x):
        """x shape (B, C, D, H, W)"""
        x = self.forward_features(x)

        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def _init_nest_weights(module: nn.Module, name: str = "", head_bias: float = 0.0):
    """NesT weight initialization
    Can replicate Jax implementation. Otherwise follows vision_transformer.py
    """
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            trunc_normal_(module.weight, std=0.02, a=-2, b=2)
            nn.init.constant_(module.bias, head_bias)
        else:
            trunc_normal_(module.weight, std=0.02, a=-2, b=2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=0.02, a=-2, b=2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def resize_pos_embed(posemb, posemb_new):
    """
    Rescale the grid of position embeddings when loading from state_dict
    Expected shape of position embeddings is (1, T, N, C), and considers only square images
    """
    _logger.info("Resized position embedding: %s to %s", posemb.shape, posemb_new.shape)
    seq_length_old = posemb.shape[2]
    num_blocks_new, seq_length_new = posemb_new.shape[1:3]
    size_new = int(math.sqrt(num_blocks_new * seq_length_new))
    # First change to (1, C, H, W)
    posemb = deblockify(posemb, int(math.sqrt(seq_length_old))).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=[size_new, size_new], mode="bicubic", align_corners=False)
    # Now change to new (1, T, N, C)
    posemb = blockify(posemb.permute(0, 2, 3, 1), int(math.sqrt(seq_length_new)))
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """resize positional embeddings of pretrained weights"""
    pos_embed_keys = [k for k in state_dict.keys() if k.startswith("pos_embed_")]
    for k in pos_embed_keys:
        if state_dict[k].shape != getattr(model, k).shape:
            state_dict[k] = resize_pos_embed(state_dict[k], getattr(model, k))
    return state_dict
