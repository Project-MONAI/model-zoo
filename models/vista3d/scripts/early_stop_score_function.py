import os

import torch
import torch.distributed as dist


def score_function(engine):
    val_metric = engine.state.metrics["val_mean_dice"]
    if dist.is_initialized():
        device = torch.device("cuda:" + os.environ["LOCAL_RANK"])
        val_metric = torch.tensor([val_metric]).to(device)
        dist.all_reduce(val_metric, op=dist.ReduceOp.SUM)
        val_metric /= dist.get_world_size()
        return val_metric.item()
    return val_metric
