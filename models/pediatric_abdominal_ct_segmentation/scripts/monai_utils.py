from __future__ import annotations

import logging
import os
from collections.abc import Hashable, Mapping
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn
from ignite.engine import Engine
from ignite.metrics import Metric
from monai.config import KeysCollection
from monai.engines import SupervisedTrainer
from monai.engines.utils import get_devices_spec
from monai.inferers import Inferer
from monai.transforms.transform import MapTransform, Transform
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim.optimizer import Optimizer

# measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

logger = logging.getLogger(__name__)

# distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")


def get_device_list(n_gpu):
    if type(n_gpu) is not list:
        n_gpu = [n_gpu]
    device_list = get_devices_spec(n_gpu)
    if torch.cuda.device_count() >= max(n_gpu):
        device_list = [d for d in device_list if d in n_gpu]
    else:
        logging.info(
            """Highest GPU ID provided in 'n_gpu' is larger than number of GPUs available, assigning GPUs starting from 0
                 to match n_gpu length of {}""".format(
                len(n_gpu)
            )
        )
        device_list = device_list[: len(n_gpu)]
    return device_list


def supervised_trainer_multi_gpu(
    max_epochs: int,
    train_data_loader,
    network: torch.nn.Module,
    optimizer: Optimizer,
    loss_function: Callable,
    device: Sequence[str | torch.device] | None = None,
    epoch_length: int | None = None,
    non_blocking: bool = False,
    iteration_update: Callable[[Engine, Any], Any] | None = None,
    inferer: Inferer | None = None,
    postprocessing: Transform | None = None,
    key_train_metric: dict[str, Metric] | None = None,
    additional_metrics: dict[str, Metric] | None = None,
    train_handlers: Sequence | None = None,
    amp: bool = False,
    distributed: bool = False,
):
    devices_ = device
    if not device:
        devices_ = get_devices_spec(device)  # Using all devices i.e GPUs

    #     if device:
    #         if next(network.parameters()).device.index != device[0]:
    #             network.to(devices_[0])
    #     else:
    #         if next(network.parameters()).device.index != devices_[0].index:
    #             network.to(devices_[0])
    #
    net = network
    if distributed:
        if len(devices_) > 1:
            raise ValueError(f"for distributed training, `devices` must contain only 1 GPU or CPU, but got {devices_}.")
        net = DistributedDataParallel(net, device_ids=devices_)
    elif len(devices_) > 1:
        net = DataParallel(net, device_ids=devices_)  # ,output_device=devices_[0])

    return SupervisedTrainer(
        device=devices_[0],
        network=net,
        optimizer=optimizer,
        loss_function=loss_function,
        max_epochs=max_epochs,
        train_data_loader=train_data_loader,
        epoch_length=epoch_length,
        non_blocking=non_blocking,
        iteration_update=iteration_update,
        inferer=inferer,
        postprocessing=postprocessing,
        key_train_metric=key_train_metric,
        additional_metrics=additional_metrics,
        train_handlers=train_handlers,
        amp=amp,
    )


class SupervisedTrainerMGPU(SupervisedTrainer):
    def __init__(
        self,
        max_epochs: int,
        train_data_loader,
        network: torch.nn.Module,
        optimizer: Optimizer,
        loss_function: Callable,
        device: Sequence[str | torch.device] | None = None,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_train_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        train_handlers: Sequence | None = None,
        amp: bool = False,
        distributed: bool = False,
    ):
        self.devices_ = device
        if not device:
            self.devices_ = get_devices_spec(device)  # Using all devices i.e GPUs

        #     if device:
        #         if next(network.parameters()).device.index != device[0]:
        #             network.to(devices_[0])
        #     else:
        #         if next(network.parameters()).device.index != devices_[0].index:
        #             network.to(devices_[0])
        #
        self.net = network
        if distributed:
            if len(self.devices_) > 1:
                raise ValueError(
                    f"for distributed training, `devices` must contain only 1 GPU or CPU, but got {self.devices_}."
                )
            self.net = DistributedDataParallel(self.net, device_ids=self.devices_)
        elif len(self.devices_) > 1:
            self.net = DataParallel(self.net, device_ids=self.devices_)  # ,output_device=devices_[0])

        super().__init__(
            device=self.devices_[0],
            network=self.net,
            optimizer=optimizer,
            loss_function=loss_function,
            max_epochs=max_epochs,
            train_data_loader=train_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            iteration_update=iteration_update,
            inferer=inferer,
            postprocessing=postprocessing,
            key_train_metric=key_train_metric,
            additional_metrics=additional_metrics,
            train_handlers=train_handlers,
            amp=amp,
        )


class AddLabelNamesd(MapTransform):
    def __init__(
        self, keys: KeysCollection, label_names: dict[str, int] | None = None, allow_missing_keys: bool = False
    ):
        """
        Normalize label values according to label names dictionary

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            label_names: all label names
        """
        super().__init__(keys, allow_missing_keys)

        self.label_names = label_names or {}

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        d["label_names"] = self.label_names
        return d


class CopyFilenamesd(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        """
        Copy Filenames for future use

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        d["filename"] = os.path.basename(d["label"])
        return d


class SplitPredsLabeld(MapTransform):
    """
    Split preds and labels for individual evaluation

    """

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            if key == "pred":
                for idx, (key_label, _) in enumerate(d["label_names"].items()):
                    if key_label != "background":
                        d[f"pred_{key_label}"] = d[key][idx, ...][None]
                        d[f"label_{key_label}"] = d["label"][idx, ...][None]
            elif key != "pred":
                logger.info("This is only for pred key")
        return d
