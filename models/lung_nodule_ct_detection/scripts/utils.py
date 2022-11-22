from typing import Dict, List, Union

import numpy as np
import torch


def detach_to_numpy(data: Union[List, Dict, torch.Tensor]) -> Union[List, Dict, torch.Tensor]:
    """
    Recursively detach elements in data
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()  # pytype: disable=attribute-error

    elif isinstance(data, np.ndarray):
        return data

    elif isinstance(data, list):
        return [detach_to_numpy(d) for d in data]

    elif isinstance(data, dict):
        for k in data.keys():
            data[k] = detach_to_numpy(data[k])
        return data

    else:
        raise ValueError("data should be tensor, numpy array, dict, or list.")
