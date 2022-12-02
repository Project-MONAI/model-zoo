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

from monai.inferers.inferer import SlidingWindowInferer, Inferer
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union, List
from torch import Tensor
import numpy as np
import torch

class RetinaNetInferer(Inferer):
    """
    RetinaNet Inferer takes RetinaNet as input

    Args:
        detector: the RetinaNetDetector that converts network output BxCxMxN or BxCxMxNxP map into boxes and classification scores.
        args: other optional args to be passed to detector.
        kwargs: other optional keyword args to be passed to detector.
    """

    def __init__(self,
        detector: RetinaNetDetector,
        *args, **kwargs
    ) -> None:
        Inferer.__init__(self)
        self.detector = detector
        self.sliding_window_size = None
        if self.detector.inferer is not None:
            try:
                self.sliding_window_size = np.prod(self.detector.inferer.roi_size)
            except:
                pass

    def __call__(self, inputs: Union[List[Tensor], Tensor], network: torch.nn.Module, *args: Any, **kwargs: Any):
        """Unified callable function API of Inferers.
        Args:
            inputs: model input data for inference.
            network: target detection network to execute inference.
                supports callable that fullfilles requirements of network in monai.apps.detection.networks.retinanet_detector.RetinaNetDetector``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.
        """
        self.detector.network = network
        self.detector.training = self.detector.network.training

        if self.sliding_window_size is not None:
            # if image smaller than sliding window roi size, no need to use sliding window inferer
            # use sliding window inferer only when image is large
            use_inferer = not all([data_i[0, ...].numel() < self.sliding_window_size for data_i in inputs])
        else:
            use_inferer = False

        return self.detector(inputs, use_inferer=use_inferer, *args, **kwargs)
