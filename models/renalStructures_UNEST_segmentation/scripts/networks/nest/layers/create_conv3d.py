#!/usr/bin/env python3

""" Create Conv2d Factory Method

Hacked together by / Copyright 2020 Ross Wightman
"""

from .conv3d_same import create_conv3d_pad


def create_conv3d(in_channels, out_channels, kernel_size, **kwargs):
    """Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv3d, or CondConv2d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """

    depthwise = kwargs.pop("depthwise", False)
    # for DW out_channels must be multiple of in_channels as must have out_channels % groups == 0
    groups = in_channels if depthwise else kwargs.pop("groups", 1)

    m = create_conv3d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m
