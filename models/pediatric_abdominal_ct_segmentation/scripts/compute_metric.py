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

"""
This example shows how to efficiently compute Dice scores for pairs of segmentation prediction
and references in multi-processing based on MONAI's metrics API.
It can even run on multi-nodes.
Main steps to set up the distributed data parallel:

- Execute `torchrun` to create processes on every node for every process.
  It receives parameters as below:
  `--nproc_per_node=NUM_PROCESSES_PER_NODE`
  `--nnodes=NUM_NODES`
  For more details, refer to https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py.
  Alternatively, we can also use `torch.multiprocessing.spawn` to start program, but it that case, need to handle
  all the above parameters and compute `rank` manually, then set to `init_process_group`, etc.
  `torchrun` is even more efficient than `torch.multiprocessing.spawn`.
- Use `init_process_group` to initialize every process.
- Partition the saved predictions and labels into ranks for parallel computation.
- Compute `Dice Metric` on every process, reduce the results after synchronization.

Note:
    `torchrun` will launch `nnodes * nproc_per_node = world_size` processes in total.
    Example script to execute this program on a single node with 2 processes:
    `torchrun --nproc_per_node=2 compute_metric.py`

Referring to: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

"""

import os

import torch
import torch.distributed as dist
from monai.data import partition_dataset
from monai.handlers import write_metrics_reports
from monai.metrics import DiceMetric
from monai.transforms import (
    AddLabelNamesd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ToDeviced,
)
from monai.utils import string_list_all_gather
from scripts.monai_utils import CopyFilenamesd


def compute(datalist, output_dir):
    # generate synthetic data for the example
    local_rank = int(os.environ["LOCAL_RANK"])
    # initialize the distributed evaluation process, change to gloo backend if computing on CPU
    dist.init_process_group(backend="nccl", init_method="env://")

    # split data for every subprocess, for example, 16 processes compute in parallel
    data_part = partition_dataset(
        data=datalist, num_partitions=dist.get_world_size(), shuffle=False, even_divisible=False
    )[dist.get_rank()]

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    # define transforms for predictions and labels
    #     labels = {'background': 0, 'liver': 1, 'spleen': 2, 'pancreas': 3}
    transforms = Compose(
        [
            CopyFilenamesd(keys="label"),
            LoadImaged(keys=["pred", "label"]),
            ToDeviced(keys=["pred", "label"], device=device),
            EnsureChannelFirstd(keys=["pred", "label"]),
            Orientationd(keys=("pred", "label"), axcodes="RAS"),
            AsDiscreted(keys=("pred", "label"), argmax=(False, False), to_onehot=(4, 4)),
        ]
    )

    data_part = [transforms(item) for item in data_part]

    # compute metrics for current process
    metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    metric(y_pred=[i["pred"] for i in data_part], y=[i["label"] for i in data_part])
    filenames = [item["filename"] for item in data_part]
    # all-gather results from all the processes and reduce for final result
    result = metric.aggregate().item()
    filenames = string_list_all_gather(strings=filenames)

    if local_rank == 0:
        print("mean dice: ", result)
        # generate metrics reports at: output/mean_dice_raw.csv, output/mean_dice_summary.csv, output/metrics.csv
        write_metrics_reports(
            save_dir=output_dir,
            images=filenames,
            metrics={"mean_dice": result},
            metric_details={"mean_dice": metric.get_buffer()},
            summary_ops="*",
        )

    metric.reset()

    dist.destroy_process_group()


def compute_single_node(datalist, output_dir):
    local_rank = int(os.environ["LOCAL_RANK"])

    filenames = [d["label"].split("/")[-1] for d in datalist]

    data_part = datalist
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # define transforms for predictions and labels
    labels = {"background": 0, "liver": 1, "spleen": 2, "pancreas": 3}
    transforms = Compose(
        [
            LoadImaged(keys=["pred", "label"]),
            ToDeviced(keys=["pred", "label"], device=device),
            EnsureChannelFirstd(keys=["pred", "label"]),
            Orientationd(keys=("pred", "label"), axcodes="RAS"),
            AddLabelNamesd(keys=("pred", "label"), label_names=labels),
            AsDiscreted(keys=("pred", "label"), argmax=(False, False), to_onehot=(4, 4)),
        ]
    )
    data_part = [transforms(item) for item in data_part]
    # compute metrics for current process
    metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    for d in datalist:
        d = transforms(d)
        metric(y_pred=[d["pred"]], y=[d["label"]])

    result = metric.aggregate().item()

    print("mean dice: ", result)
    write_metrics_reports(
        save_dir=output_dir,
        images=filenames,
        metrics={"mean_dice": result},
        metric_details={"mean_dice": metric.get_buffer()},
        summary_ops="*",
    )

    metric.reset()
