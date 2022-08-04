#!/usr/bin/env python3

""" Bilinear-Attention-Transform and Non-Local Attention

Paper: `Non-Local Neural Networks With Grouped Bilinear Attentional Transforms`
Adapted from original code: https://github.com/BA-Transform/BAT-Image-Classification
"""
import torch
from torch import nn
from torch.nn import functional as F

from .conv_bn_act import ConvBnAct
from .helpers import make_divisible
from .trace_utils import _assert


class NonLocalAttn(nn.Module):
    """Spatial NL block for image classification.

    This was adapted from https://github.com/BA-Transform/BAT-Image-Classification
    Their NonLocal impl inspired by https://github.com/facebookresearch/video-nonlocal-net.
    """

    def __init__(self, in_channels, use_scale=True, rd_ratio=1 / 8, rd_channels=None, rd_divisor=8, **kwargs):
        super(NonLocalAttn, self).__init__()
        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        self.scale = in_channels**-0.5 if use_scale else 1.0
        self.t = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.p = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.g = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.z = nn.Conv2d(rd_channels, in_channels, kernel_size=1, stride=1, bias=True)
        self.norm = nn.BatchNorm2d(in_channels)
        self.reset_parameters()

    def forward(self, x):
        shortcut = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()
        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)

        att = torch.bmm(t, p) * self.scale
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, g)

        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        x = self.z(x)
        x = self.norm(x) + shortcut

        return x

    def reset_parameters(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if len(list(m.parameters())) > 1:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)


class BilinearAttnTransform(nn.Module):
    def __init__(self, in_channels, block_size, groups, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(BilinearAttnTransform, self).__init__()

        self.conv1 = ConvBnAct(in_channels, groups, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.conv_p = nn.Conv2d(groups, block_size * block_size * groups, kernel_size=(block_size, 1))
        self.conv_q = nn.Conv2d(groups, block_size * block_size * groups, kernel_size=(1, block_size))
        self.conv2 = ConvBnAct(in_channels, in_channels, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.block_size = block_size
        self.groups = groups
        self.in_channels = in_channels

    def resize_mat(self, x, t: int):
        b, c, block_size, block_size1 = x.shape
        _assert(block_size == block_size1, "")
        if t <= 1:
            return x
        x = x.view(b * c, -1, 1, 1)
        x = x * torch.eye(t, t, dtype=x.dtype, device=x.device)
        x = x.view(b * c, block_size, block_size, t, t)
        x = torch.cat(torch.split(x, 1, dim=1), dim=3)
        x = torch.cat(torch.split(x, 1, dim=2), dim=4)
        x = x.view(b, c, block_size * t, block_size * t)
        return x

    def forward(self, x):
        _assert(x.shape[-1] % self.block_size == 0, "")
        _assert(x.shape[-2] % self.block_size == 0, "")
        b, c, h, w = x.shape
        out = self.conv1(x)
        rp = F.adaptive_max_pool2d(out, (self.block_size, 1))
        cp = F.adaptive_max_pool2d(out, (1, self.block_size))
        p = self.conv_p(rp).view(b, self.groups, self.block_size, self.block_size).sigmoid()
        q = self.conv_q(cp).view(b, self.groups, self.block_size, self.block_size).sigmoid()
        p = p / p.sum(dim=3, keepdim=True)
        q = q / q.sum(dim=2, keepdim=True)
        p = (
            p.view(b, self.groups, 1, self.block_size, self.block_size)
            .expand(x.size(0), self.groups, c // self.groups, self.block_size, self.block_size)
            .contiguous()
        )
        p = p.view(b, c, self.block_size, self.block_size)
        q = (
            q.view(b, self.groups, 1, self.block_size, self.block_size)
            .expand(x.size(0), self.groups, c // self.groups, self.block_size, self.block_size)
            .contiguous()
        )
        q = q.view(b, c, self.block_size, self.block_size)
        p = self.resize_mat(p, h // self.block_size)
        q = self.resize_mat(q, w // self.block_size)
        y = p.matmul(x)
        y = y.matmul(q)

        y = self.conv2(y)
        return y


class BatNonLocalAttn(nn.Module):
    """BAT
    Adapted from: https://github.com/BA-Transform/BAT-Image-Classification
    """

    def __init__(
        self,
        in_channels,
        block_size=7,
        groups=2,
        rd_ratio=0.25,
        rd_channels=None,
        rd_divisor=8,
        drop_rate=0.2,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        **_,
    ):
        super().__init__()
        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        self.conv1 = ConvBnAct(in_channels, rd_channels, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.ba = BilinearAttnTransform(rd_channels, block_size, groups, act_layer=act_layer, norm_layer=norm_layer)
        self.conv2 = ConvBnAct(rd_channels, in_channels, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.dropout = nn.Dropout2d(p=drop_rate)

    def forward(self, x):
        xl = self.conv1(x)
        y = self.ba(xl)
        y = self.conv2(y)
        y = self.dropout(y)
        return y + x
