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

import copy
from typing import List, Union

import torch
from monai.apps.vista3d.inferer import point_based_window_inferer
from monai.inferers import Inferer, SlidingWindowInfererAdapt
from torch import Tensor


class Vista3dInferer(Inferer):
    """
    Vista3D Inferer

    Args:
        roi_size: the sliding window patch size.
        overlap: sliding window overlap ratio.
    """

    def __init__(self, roi_size, overlap, use_point_window=False, sw_batch_size=1) -> None:
        Inferer.__init__(self)
        self.roi_size = roi_size
        self.overlap = overlap
        self.sw_batch_size = sw_batch_size
        self.use_point_window = use_point_window

    def __call__(
        self,
        inputs: Union[List[Tensor], Tensor],
        network,
        point_coords,
        point_labels,
        class_vector,
        labels=None,
        label_set=None,
        prev_mask=None,
    ):
        """
        Unified callable function API of Inferers.
        Notice: The point_based_window_inferer currently only supports SINGLE OBJECT INFERENCE with B=1.
        It only used in interactive segmentation.

        Args:
            inputs: input tensor images.
            network: vista3d model.
            point_coords: point click coordinates. [B, N, 3].
            point_labels: point click labels (0 for negative, 1 for positive) [B, N].
            class_vector: class vector of length B.
            labels: groundtruth labels. Used for sampling validation points.
            label_set: [0,1,2,3,...,output_classes].
            prev_mask: [1, B, H, W, D], THE VALUE IS BEFORE SIGMOID!

        """
        prompt_class = copy.deepcopy(class_vector)
        if class_vector is not None:
            # Check if network has attribute 'point_head' directly or within its 'module'
            if hasattr(network, "point_head"):
                point_head = network.point_head
            elif hasattr(network, "module") and hasattr(network.module, "point_head"):
                point_head = network.module.point_head
            else:
                raise AttributeError("Network does not have attribute 'point_head'.")

            if torch.any(class_vector > point_head.last_supported):
                class_vector = None
        val_outputs = None
        torch.cuda.empty_cache()
        if self.use_point_window and point_coords is not None:
            if isinstance(inputs, list):
                device = inputs[0].device
            else:
                device = inputs.device
            val_outputs = point_based_window_inferer(
                inputs=inputs,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                transpose=True,
                with_coord=True,
                predictor=network,
                mode="gaussian",
                sw_device=device,
                device=device,
                overlap=self.overlap,
                point_coords=point_coords,
                point_labels=point_labels,
                class_vector=class_vector,
                prompt_class=prompt_class,
                prev_mask=prev_mask,
                labels=labels,
                label_set=label_set,
            )
        else:
            val_outputs = SlidingWindowInfererAdapt(
                roi_size=self.roi_size, sw_batch_size=self.sw_batch_size, with_coord=True, padding_mode="replicate"
            )(
                inputs,
                network,
                transpose=True,
                point_coords=point_coords,
                point_labels=point_labels,
                class_vector=class_vector,
                prompt_class=prompt_class,
                prev_mask=prev_mask,
                labels=labels,
                label_set=label_set,
            )
        return val_outputs
