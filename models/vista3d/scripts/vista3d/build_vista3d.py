#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .modeling import VISTA3D2, ClassMappingClassify, PointMappingSAM, SegResNetDS2


def build_vista3d_segresnet_decoder(encoder_embed_dim=48, in_channels=1, image_size=(96, 96, 96)):
    segresnet = SegResNetDS2(
        in_channels=in_channels,
        blocks_down=(1, 2, 2, 4, 4),
        norm="instance",
        out_channels=encoder_embed_dim,
        init_filters=encoder_embed_dim,
        dsdepth=1,
    )
    point_head = PointMappingSAM(feature_size=encoder_embed_dim, n_classes=512, last_supported=132)
    class_head = ClassMappingClassify(n_classes=512, feature_size=encoder_embed_dim, use_mlp=True)
    vista = VISTA3D2(
        image_encoder=segresnet, class_head=class_head, point_head=point_head, feature_size=encoder_embed_dim
    )
    return vista


vista_model_registry = {"vista3d_segresnet_d": build_vista3d_segresnet_decoder}
