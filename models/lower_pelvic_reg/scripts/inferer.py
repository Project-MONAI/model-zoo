from typing import Dict, Callable, Any, Tuple

import torch
from monai.inferers import Inferer
from monai.networks import one_hot
from monai.networks.blocks import Warp


class RegistrationInferer(Inferer):
    def __init__(self) -> None:
        Inferer.__init__(self)
        self.warp = Warp()

    def __call__(self, input: Tuple[Dict, Dict], network: Callable[..., torch.Tensor], *args: Any, **kwargs: Any):
        """Unified callable function API of Inferers.

        Args:
            input: a pair of dictionaries specifying moving and fixed, both dictionaries are expected to have keys
            "image" and "label" with values both of shape (B, C, ...)
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.
        """
        moving, fixed = input
        ddf = network(
            torch.cat([moving["image"], fixed["image"]], dim=1)
        )
        warped_image = self.warp(image=moving["image"], ddf=ddf)
        moving_label_onehot = one_hot(
            moving["label"],
            num_classes=int(max(torch.unique(moving["label"])) + 1)
        )
        warped_label_onehot = self.warp(image=moving_label_onehot, ddf=ddf)
        warped_label = torch.argmax(warped_label_onehot, dim=1)
        output = {
            "ddf": ddf,
            "warped_image": warped_image,
            "warped_label": warped_label
        }
        return output
