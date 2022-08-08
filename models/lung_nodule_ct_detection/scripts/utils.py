from typing import Callable, Sequence, Union, Dict, List
import torch
import numpy as np

def detach_to_numpy(data: Union[List, Dict, torch.Tensor]) -> Union[List, Dict, torch.Tensor]:
    """
    Recursively detach elements in data
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()

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