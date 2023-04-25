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

from typing import Any, List, Union

import numpy as np
import torch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.inferers.inferer import Inferer
from torch import Tensor


class RetinaNetInferer(Inferer):
    """
    RetinaNet Inferer takes RetinaNet as input

    Args:
        detector: the RetinaNetDetector that converts network output BxCxMxN or BxCxMxNxP
            map into boxes and classification scores.
        force_sliding_window: whether to force using a SlidingWindowInferer to do the inference.
                If False, will check the input spatial size to decide whether to simply
                forward the network or using SlidingWindowInferer.
                If True, will force using SlidingWindowInferer to do the inference.
        args: other optional args to be passed to detector.
        kwargs: other optional keyword args to be passed to detector.
    """

    def __init__(self, detector: RetinaNetDetector, force_sliding_window: bool = False) -> None:
        Inferer.__init__(self)
        self.detector = detector
        self.sliding_window_size = None
        self.force_sliding_window = force_sliding_window
        if self.detector.inferer is not None:
            if hasattr(self.detector.inferer, "roi_size"):
                self.sliding_window_size = np.prod(self.detector.inferer.roi_size)

    def __call__(self, inputs: Union[List[Tensor], Tensor], network: torch.nn.Module, *args: Any, **kwargs: Any):
        """Unified callable function API of Inferers.
        Args:
            inputs: model input data for inference.
            network: target detection network to execute inference.
                supports callable that fullfilles requirements of network in
                monai.apps.detection.networks.retinanet_detector.RetinaNetDetector``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.
        """
        self.detector.network = network
        self.detector.training = self.detector.network.training

        # if image smaller than sliding window roi size, no need to use sliding window inferer
        # use sliding window inferer only when image is large
        use_inferer = (
            self.force_sliding_window
            or self.sliding_window_size is not None
            and not all([data_i[0, ...].numel() < self.sliding_window_size for data_i in inputs])
        )

        return self.detector(inputs, use_inferer=use_inferer, *args, **kwargs)
