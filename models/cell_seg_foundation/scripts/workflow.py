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

import csv
import gc
import logging
import os
import shutil
import sys
import time
from collections import OrderedDict
from datetime import datetime

import monai.transforms as mt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from monai.apps import get_logger
from monai.auto3dseg.utils import datafold_read
from monai.bundle import BundleWorkflow, ConfigParser
from monai.config import print_config
from monai.data import DataLoader, Dataset, decollate_batch
from monai.metrics import CumulativeAverage
from monai.utils import (
    BundleProperty,
    ImageMetaKey,
    convert_to_dst_type,
    ensure_tuple,
    look_up_option,
    optional_import,
    set_determinism,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

mlflow, mlflow_is_imported = optional_import("mlflow")


if __package__ in (None, ""):
    from cell_distributed_weighted_sampler import DistributedWeightedSampler
    from components import LabelsToFlows, LoadTiffd, LogitsToLabels
    from utils import LOGGING_CONFIG, parsing_bundle_config  # type: ignore
else:
    from .cell_distributed_weighted_sampler import DistributedWeightedSampler
    from .components import LabelsToFlows, LoadTiffd, LogitsToLabels
    from .utils import LOGGING_CONFIG, parsing_bundle_config


logger = get_logger("VistaCell")


class VistaCell(BundleWorkflow):
    """
    Primary vista model training workflow that extends
    monai.bundle.BundleWorkflow for cell segmentation.
    """

    def __init__(
        self,
        config_file=None,
        meta_file=None,
        logging_file=None,
        workflow_type="train",
        **override,
    ):
        """
        config_file can be one or a list of config files.
        the rest key-values in the `override` are to override config content.
        """

        parser = parsing_bundle_config(config_file, logging_file=logging_file, meta_file=meta_file)
        parser.update(pairs=override)

        mode = parser.get("mode", None)
        if mode is not None:  # if user specified a `mode` it'll override the workflow_type arg
            workflow_type = mode
        else:
            mode = workflow_type  # if user didn't specify mode, the workflow_type will be used
        super().__init__(workflow_type=workflow_type)
        self._props = {}
        self._set_props = {}
        self.parser = parser

        self.rank = int(os.getenv("LOCAL_RANK", "0"))
        self.global_rank = int(os.getenv("RANK", "0"))
        self.is_distributed = dist.is_available() and dist.is_initialized()

        # check if torchrun or bcprun started it
        if dist.is_torchelastic_launched() or (
            os.getenv("NGC_ARRAY_SIZE") is not None and int(os.getenv("NGC_ARRAY_SIZE")) > 1
        ):
            if dist.is_available():
                dist.init_process_group(backend="nccl", init_method="env://")

            self.is_distributed = dist.is_available() and dist.is_initialized()

            torch.cuda.set_device(self.config("device"))
            dist.barrier()

        else:
            self.is_distributed = False

        if self.global_rank == 0 and self.config("ckpt_path") and not os.path.exists(self.config("ckpt_path")):
            os.makedirs(self.config("ckpt_path"), exist_ok=True)

        if self.rank == 0:
            # make sure the log file exists, as a workaround for mult-gpu logging race condition
            _log_file = self.config("log_output_file", "vista_cell.log")
            if _log_file is not None:
                open(_log_file, "a").close()

            print_config()

        if self.is_distributed:
            dist.barrier()

        seed = self.config("seed", None)
        if seed is not None:
            set_determinism(seed)
            logger.info(f"set determinism seed: {self.config('seed', None)}")
        elif torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("No seed provided, using cudnn.benchmark for performance.")

        if os.path.exists(self.config("ckpt_path")):
            self.parser.export_config_file(
                self.parser.config,
                os.path.join(self.config("ckpt_path"), "working.yaml"),
                fmt="yaml",
                default_flow_style=None,
            )

        self.add_property("network", required=True)
        self.add_property("train_loader", required=True)
        self.add_property("val_dataset", required=False)
        self.add_property("val_loader", required=False)
        self.add_property("val_preprocessing", required=False)
        self.add_property("train_sampler", required=True)
        self.add_property("val_sampler", required=True)
        self.add_property("mode", required=False)
        # set evaluator as required when mode is infer or eval
        # will change after we enhance the bundle properties
        self.evaluator = None

    def _set_property(self, name, property, value):
        # stores user-reset initialized objects that should not be re-initialized.
        self._set_props[name] = value

    def _get_property(self, name, property):
        """
        The customized bundle workflow must implement required properties in:
        https://github.com/Project-MONAI/MONAI/blob/dev/monai/bundle/properties.py.
        """
        if name in self._set_props:
            self._props[name] = self._set_props[name]
            return self._props[name]
        if name in self._props:
            return self._props[name]
        try:
            value = getattr(self, f"get_{name}")()
        except AttributeError:
            if property[BundleProperty.REQUIRED]:
                raise ValueError(
                    f"Property '{name}' is required by the bundle format,"
                    f"but the method 'get_{name}' is not implemented."
                )
            raise AttributeError
        self._props[name] = value
        return value

    def config(self, name, default="null", **kwargs):
        """read the parsed content (evaluate the expression) from the config file."""
        if default != "null":
            return self.parser.get_parsed_content(name, default=default, **kwargs)
        return self.parser.get_parsed_content(name, **kwargs)

    def initialize(self):
        _log_file = self.config("log_output_file", "vista_cell.log")
        if _log_file is None:
            LOGGING_CONFIG["loggers"]["VistaCell"]["handlers"].remove("file")
            LOGGING_CONFIG["handlers"].pop("file", None)
        else:
            LOGGING_CONFIG["handlers"]["file"]["filename"] = _log_file
        logging.config.dictConfig(LOGGING_CONFIG)

    def get_mode(self):
        mode_str = self.config("mode", self.workflow_type)
        return look_up_option(mode_str, ("train", "training", "infer", "inference", "eval", "evaluation"))

    def run(self):
        if str(self.mode).startswith("train"):
            return self.train()
        if str(self.mode).startswith("infer"):
            return self.infer()
        return self.validate()

    def finalize(self):
        if self.is_distributed:
            dist.destroy_process_group()
        set_determinism(None)

    def get_network_def(self):
        return self.config("network_def")

    def get_network(self):
        pretrained_ckpt_name = self.config("pretrained_ckpt_name", None)
        pretrained_ckpt_path = self.config("pretrained_ckpt_path", None)
        if pretrained_ckpt_name is not None and pretrained_ckpt_path is None:
            # if relative name specified, append to default ckpt_path dir
            pretrained_ckpt_path = os.path.join(self.config("ckpt_path"), pretrained_ckpt_name)

        if pretrained_ckpt_path is not None and not os.path.exists(pretrained_ckpt_path):
            logger.info(f"Pretrained checkpoint {pretrained_ckpt_path} not found.")
            raise ValueError(f"Pretrained checkpoint {pretrained_ckpt_path} not found.")

        if pretrained_ckpt_path is not None and os.path.exists(pretrained_ckpt_path):
            # not loading sam weights, if we're using our own checkpoint
            if "checkpoint" in self.parser.config["network_def"]:
                self.parser.config["network_def"]["checkpoint"] = None
            model = self.config("network")
            self.checkpoint_load(ckpt=pretrained_ckpt_path, model=model)
        else:
            model = self.config("network")

        if self.config("channels_last", False):
            model = model.to(memory_format=torch.channels_last)

        if self.is_distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if self.config("compile", False):
            model = torch.compile(model)

        if self.is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=self.config("find_unused_parameters", False),
            )

        pytorch_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"total parameters count {pytorch_params} distributed {self.is_distributed}")
        return model

    def get_train_dataset_data(self):
        train_files, valid_files = [], []
        dataset_data = self.config("train#dataset#data")
        val_key = None
        if isinstance(dataset_data, dict):
            val_key = dataset_data.get("key", None)
        data_list_files = dataset_data["data_list_files"]

        if isinstance(data_list_files, str):
            data_list_files = ConfigParser.load_config_file(
                data_list_files
            )  # if it's a path to a separate file with a list of datasets
        else:
            data_list_files = ensure_tuple(data_list_files)

        if self.global_rank == 0:
            print("Using data_list_files ", data_list_files)

        for idx, d in enumerate(data_list_files):
            logger.info(f"adding datalist ({idx}): {d['datalist']}")
            t, v = datafold_read(datalist=d["datalist"], basedir=d["basedir"], fold=self.config("fold"))

            if val_key is not None:
                v, _ = datafold_read(datalist=d["datalist"], basedir=d["basedir"], fold=-1, key=val_key)  # e.g. testing

            for item in t:
                item["datalist_id"] = idx
                item["datalist_count"] = len(t)
            for item in v:
                item["datalist_id"] = idx
                item["datalist_count"] = len(v)
            train_files.extend(t)
            valid_files.extend(v)

        if self.config("quick", False):
            logger.info("quick_data")
            train_files = train_files[:8]
            valid_files = valid_files[:7]
        if not valid_files:
            logger.warning("No validation data found.")
        return train_files, valid_files

    def read_val_datalists(self, section="validate", data_list_files=None, val_key=None, merge=True):
        """read the corresponding folds of the datalist for validation or inference"""
        dataset_data = self.config(f"{section}#dataset#data")

        if isinstance(dataset_data, list):
            return dataset_data

        if data_list_files is None:
            data_list_files = dataset_data["data_list_files"]

        if isinstance(data_list_files, str):
            data_list_files = ConfigParser.load_config_file(
                data_list_files
            )  # if it's a path to a separate file with a list of datasets
        else:
            data_list_files = ensure_tuple(data_list_files)

        if val_key is None:
            val_key = dataset_data.get("key", None)

        val_files, idx = [], 0
        for d in data_list_files:
            if val_key is not None:
                v_files, _ = datafold_read(datalist=d["datalist"], basedir=d["basedir"], fold=-1, key=val_key)
            else:
                _, v_files = datafold_read(datalist=d["datalist"], basedir=d["basedir"], fold=self.config("fold"))
            logger.info(f"adding datalist ({idx} -- {val_key}): {d['datalist']} {len(v_files)}")
            if merge:
                val_files.extend(v_files)
            else:
                val_files.append(v_files)
            idx += 1

        if self.config("quick", False):
            logger.info("quick_data")
            val_files = val_files[:8] if merge else [val_files[0][:8]]
        return val_files

    def get_train_preprocessing(self):
        roi_size = self.config("train#dataset#preprocessing#roi_size")

        train_xforms = []
        train_xforms.append(LoadTiffd(keys=["image", "label"]))
        train_xforms.append(mt.EnsureTyped(keys=["image", "label"], data_type="tensor", dtype=torch.float))
        if self.config("prescale", True):
            print("Prescaling images to 0..1")
            train_xforms.append(mt.ScaleIntensityd(keys="image", minv=0, maxv=1, channel_wise=True))
        train_xforms.append(mt.ScaleIntensityd(keys="image", minv=0, maxv=1, channel_wise=True))
        train_xforms.append(
            mt.ScaleIntensityRangePercentilesd(
                keys="image", lower=1, upper=99, b_min=0.0, b_max=1.0, channel_wise=True, clip=True
            )
        )
        train_xforms.append(mt.SpatialPadd(keys=["image", "label"], spatial_size=roi_size))
        train_xforms.append(
            mt.RandSpatialCropd(keys=["image", "label"], roi_size=roi_size)
        )  # crop roi_size (if image is large)

        # # add augmentations
        train_xforms.extend(
            [
                mt.RandAffined(
                    keys=["image", "label"],
                    prob=0.5,
                    rotate_range=np.pi,  # from -pi to pi
                    scale_range=[-0.5, 0.5],  # from 0.5 to 1.5
                    mode=["bilinear", "nearest"],
                    spatial_size=roi_size,
                    cache_grid=True,
                    padding_mode="border",
                ),
                mt.RandAxisFlipd(keys=["image", "label"], prob=0.5),
                mt.RandGaussianNoised(keys=["image"], prob=0.25, mean=0, std=0.1),
                mt.RandAdjustContrastd(keys=["image"], prob=0.25, gamma=(1, 2)),
                mt.RandGaussianSmoothd(keys=["image"], prob=0.25, sigma_x=(1, 2)),
                mt.RandHistogramShiftd(keys=["image"], prob=0.25, num_control_points=3),
                mt.RandGaussianSharpend(keys=["image"], prob=0.25),
            ]
        )

        train_xforms.append(
            LabelsToFlows(keys="label", flow_key="flow")
        )  # finally create new key "flows" with 3 channels 1) foreground 2) dx flow 3) dy flow

        return train_xforms

    def get_val_preprocessing(self):
        val_xforms = []
        val_xforms.append(LoadTiffd(keys=["image", "label"], allow_missing_keys=True))
        val_xforms.append(
            mt.EnsureTyped(keys=["image", "label"], data_type="tensor", dtype=torch.float, allow_missing_keys=True)
        )

        if self.config("prescale", True):
            print("Prescaling val images to 0..1")
            val_xforms.append(mt.ScaleIntensityd(keys="image", minv=0, maxv=1, channel_wise=True))

        val_xforms.append(
            mt.ScaleIntensityRangePercentilesd(
                keys="image", lower=1, upper=99, b_min=0.0, b_max=1.0, channel_wise=True, clip=True
            )
        )
        val_xforms.append(LabelsToFlows(keys="label", flow_key="flow", allow_missing_keys=True))

        return val_xforms

    def get_train_dataset(self):
        train_dataset_data = self.config("train#dataset#data")
        if isinstance(train_dataset_data, list):  # FIXME, why check
            train_files = train_dataset_data
        else:
            train_files, _ = self.train_dataset_data
        logger.info(f"train files {len(train_files)}")
        return Dataset(data=train_files, transform=mt.Compose(self.train_preprocessing))

    def get_val_dataset(self):
        """this is to be used for validation during training"""
        val_dataset_data = self.config("validate#dataset#data")
        if isinstance(val_dataset_data, list):  # FIXME, why check
            valid_files = val_dataset_data
        else:
            _, valid_files = self.train_dataset_data
        return Dataset(data=valid_files, transform=mt.Compose(self.val_preprocessing))

    def set_val_datalist(self, datalist_py):
        self.parser["validate#dataset#data"] = datalist_py
        self._props.pop("val_loader", None)
        self._props.pop("val_dataset", None)
        self._props.pop("val_sampler", None)

    def get_train_sampler(self):
        if self.config("use_weighted_sampler", False):
            data = self.train_dataset.data
            logger.info(f"Using weighted sampler, first item {data[0]}")
            sample_weights = 1.0 / torch.as_tensor(
                [item.get("datalist_count", 1.0) for item in data], dtype=torch.float
            )  # inverse proportional to sub-datalist count
            # if we are using weighed sampling, the number of iterations epoch must be provided
            # (cant use a dataset length anymore)
            num_samples_per_epoch = self.config("num_samples_per_epoch", None)
            if num_samples_per_epoch is None:
                num_samples_per_epoch = len(data)  # a workaround if not provided
                logger.warning(
                    "We are using weighted random sampler, but num_samples_per_epoch is not provided, "
                    f"so using {num_samples_per_epoch} full data length as a workaround!"
                )

            if self.is_distributed:
                return DistributedWeightedSampler(
                    self.train_dataset, shuffle=True, weights=sample_weights, num_samples=num_samples_per_epoch
                )  # custom implementation, as Pytorch does not have one
            return WeightedRandomSampler(weights=sample_weights, num_samples=num_samples_per_epoch)

        if self.is_distributed:
            return DistributedSampler(self.train_dataset, shuffle=True)
        return None

    def get_val_sampler(self):
        if self.is_distributed:
            return DistributedSampler(self.val_dataset, shuffle=False)
        return None

    def get_train_loader(self):
        sampler = self.train_sampler
        return DataLoader(
            self.train_dataset,
            batch_size=self.config("train#batch_size"),
            shuffle=(sampler is None),
            sampler=sampler,
            pin_memory=True,
            num_workers=self.config("train#num_workers"),
        )

    def get_val_loader(self):
        sampler = self.val_sampler
        return DataLoader(
            self.val_dataset,
            batch_size=self.config("validate#batch_size"),
            shuffle=False,
            sampler=sampler,
            pin_memory=True,
            num_workers=self.config("validate#num_workers"),
        )

    def train(self):
        config = self.config
        distributed = self.is_distributed
        sliding_inferrer = config("inferer#sliding_inferer")
        use_amp = config("amp")

        amp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            config("amp_dtype")
        ]
        if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            amp_dtype = torch.float16
            logger.warning(
                "bfloat16 dtype is not support on your device, changing to float16, use  --amp_dtype=float16 to set manually"
            )

        use_gradscaler = use_amp and amp_dtype == torch.float16
        logger.info(f"Using grad scaler {use_gradscaler} amp_dtype {amp_dtype} use_amp {use_amp}")
        grad_scaler = GradScaler(enabled=use_gradscaler)  # using GradScaler only for AMP float16 (not bfloat16)

        loss_function = config("loss_function")
        acc_function = config("key_metric")

        ckpt_path = config("ckpt_path")
        channels_last = config("channels_last")

        num_epochs_per_saving = config("train#trainer#num_epochs_per_saving")
        num_epochs_per_validation = config("train#trainer#num_epochs_per_validation")
        num_epochs = config("train#trainer#max_epochs")
        val_schedule_list = self.schedule_validation_epochs(
            num_epochs=num_epochs, num_epochs_per_validation=num_epochs_per_validation
        )
        logger.info(f"Scheduling validation loops at epochs: {val_schedule_list}")

        train_loader = self.train_loader
        val_loader = self.val_loader
        optimizer = config("optimizer")
        model = self.network

        tb_writer = None
        csv_path = progress_path = None

        if self.global_rank == 0 and ckpt_path is not None:
            # rank 0 is responsible for heavy lifting of logging/saving
            progress_path = os.path.join(ckpt_path, "progress.yaml")

            tb_writer = SummaryWriter(log_dir=ckpt_path)
            logger.info(f"Writing Tensorboard logs to {tb_writer.log_dir}")

            if mlflow_is_imported:
                if config("mlflow_tracking_uri", None) is not None:
                    mlflow.set_tracking_uri(config("mlflow_tracking_uri"))
                    mlflow.set_experiment("vista2d")

                    mlflow_run_name = config("mlflow_run_name", f'vista2d train fold{config("fold")}')
                    mlflow.start_run(
                        run_name=mlflow_run_name, log_system_metrics=config("mlflow_log_system_metrics", False)
                    )
                    mlflow.log_params(self.parser.config)
                    mlflow.log_dict(self.parser.config, "hyper_parameters.yaml")  # experimental

            csv_path = os.path.join(ckpt_path, "accuracy_history.csv")
            self.save_history_csv(
                csv_path=csv_path,
                header=["epoch", "metric", "loss", "iter", "time", "train_time", "validation_time", "epoch_time"],
            )

        do_torch_save = (
            (self.global_rank == 0) and ckpt_path and config("ckpt_save") and not config("train#skip", False)
        )
        best_ckpt_path = os.path.join(ckpt_path, "model.pt")
        intermediate_ckpt_path = os.path.join(ckpt_path, "model_final.pt")

        best_metric = float(config("best_metric", -1))
        start_epoch = config("start_epoch", 0)
        best_metric_epoch = -1
        pre_loop_time = time.time()
        report_num_epochs = num_epochs
        train_time = validation_time = 0
        val_acc_history = []

        if start_epoch > 0:
            val_schedule_list = [v for v in val_schedule_list if v >= start_epoch]
            if len(val_schedule_list) == 0:
                val_schedule_list = [start_epoch]
            print(f"adjusted schedule_list {val_schedule_list}")

        logger.info(
            f"Using num_epochs => {num_epochs}\n "
            f"Using start_epoch => {start_epoch}\n "
            f"batch_size => {config('train#batch_size')} \n "
            f"num_warmup_epochs => {config('train#trainer#num_warmup_epochs')} \n "
        )

        lr_scheduler = config("lr_scheduler")
        if lr_scheduler is not None and start_epoch > 0:
            lr_scheduler.last_epoch = start_epoch

        range_num_epochs = range(start_epoch, num_epochs)

        if distributed:
            dist.barrier()

        if self.global_rank == 0 and tb_writer is not None and mlflow_is_imported and mlflow.is_tracking_uri_set():
            mlflow.log_param("len_train_set", len(train_loader.dataset))
            mlflow.log_param("len_val_set", len(val_loader.dataset))

        for epoch in range_num_epochs:
            report_epoch = epoch

            if distributed:
                if isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)
                dist.barrier()

            epoch_time = start_time = time.time()

            train_loss, train_acc = 0, 0

            if not config("train#skip", False):
                train_loss, train_acc = self.train_epoch(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    acc_function=acc_function,
                    grad_scaler=grad_scaler,
                    epoch=report_epoch,
                    rank=self.rank,
                    global_rank=self.global_rank,
                    num_epochs=report_num_epochs,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    channels_last=channels_last,
                    device=config("device"),
                )

            train_time = time.time() - start_time
            logger.info(
                f"Latest training  {report_epoch}/{report_num_epochs - 1} "
                f"loss: {train_loss:.4f} time {train_time:.2f}s  "
                f"lr: {optimizer.param_groups[0]['lr']:.4e}"
            )

            if self.global_rank == 0 and tb_writer is not None:
                tb_writer.add_scalar("train/loss", train_loss, report_epoch)

                if mlflow_is_imported and mlflow.is_tracking_uri_set():
                    mlflow.log_metric("train/loss", train_loss, step=report_epoch)
                    mlflow.log_metric("train/epoch_time", train_time, step=report_epoch)

            # validate every num_epochs_per_validation epochs (defaults to 1, every epoch)
            val_acc_mean = -1
            if (
                len(val_schedule_list) > 0
                and epoch + 1 >= val_schedule_list[0]
                and val_loader is not None
                and len(val_loader) > 0
            ):
                val_schedule_list.pop(0)

                start_time = time.time()
                torch.cuda.empty_cache()

                val_loss, val_acc = self.val_epoch(
                    model=model,
                    val_loader=val_loader,
                    sliding_inferrer=sliding_inferrer,
                    loss_function=loss_function,
                    acc_function=acc_function,
                    epoch=report_epoch,
                    rank=self.rank,
                    global_rank=self.global_rank,
                    num_epochs=report_num_epochs,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    channels_last=channels_last,
                    device=config("device"),
                )

                torch.cuda.empty_cache()
                validation_time = time.time() - start_time

                val_acc_mean = float(np.mean(val_acc))
                val_acc_history.append((report_epoch, val_acc_mean))

                if self.global_rank == 0:
                    logger.info(
                        f"Latest validation {report_epoch}/{report_num_epochs - 1} "
                        f"loss: {val_loss:.4f} acc_avg: {val_acc_mean:.4f} acc: {val_acc} time: {validation_time:.2f}s"
                    )

                    if tb_writer is not None:
                        tb_writer.add_scalar("val/acc", val_acc_mean, report_epoch)
                        tb_writer.add_scalar("val/loss", val_loss, report_epoch)
                        if mlflow_is_imported and mlflow.is_tracking_uri_set():
                            mlflow.log_metric("val/acc", val_acc_mean, step=report_epoch)
                            mlflow.log_metric("val/epoch_time", validation_time, step=report_epoch)

                    timing_dict = {
                        "time": f"{(time.time() - pre_loop_time) / 3600:.2f} hr",
                        "train_time": f"{train_time:.2f}s",
                        "validation_time": f"{validation_time:.2f}s",
                        "epoch_time": f"{time.time() - epoch_time:.2f}s",
                    }

                    if val_acc_mean > best_metric:
                        logger.info(f"New best metric ({best_metric:.6f} --> {val_acc_mean:.6f}). ")
                        best_metric, best_metric_epoch = val_acc_mean, report_epoch
                        save_time = 0
                        if do_torch_save:
                            save_time = self.checkpoint_save(
                                ckpt=best_ckpt_path, model=model, epoch=best_metric_epoch, best_metric=best_metric
                            )

                        if progress_path is not None:
                            self.save_progress_yaml(
                                progress_path=progress_path,
                                ckpt=best_ckpt_path if do_torch_save else None,
                                best_avg_score_epoch=best_metric_epoch,
                                best_avg_score=best_metric,
                                save_time=save_time,
                                **timing_dict,
                            )
                    if csv_path is not None:
                        self.save_history_csv(
                            csv_path=csv_path,
                            epoch=report_epoch,
                            metric=f"{val_acc_mean:.4f}",
                            loss=f"{train_loss:.4f}",
                            iter=report_epoch * len(train_loader.dataset),
                            **timing_dict,
                        )

                # sanity check
                if epoch > max(20, num_epochs / 4) and 0 <= val_acc_mean < 0.01 and config("stop_on_lowacc", True):
                    logger.info(
                        f"Accuracy seems very low at epoch {report_epoch}, acc {val_acc_mean}. "
                        "Most likely optimization diverged, try setting a smaller learning_rate"
                        f" than {config('learning_rate')}"
                    )
                    raise ValueError(
                        f"Accuracy seems very low at epoch {report_epoch}, acc {val_acc_mean}. "
                        "Most likely optimization diverged, try setting a smaller learning_rate"
                        f" than {config('learning_rate')}"
                    )

            # save intermediate checkpoint every num_epochs_per_saving epochs
            if do_torch_save and ((epoch + 1) % num_epochs_per_saving == 0 or (epoch + 1) >= num_epochs):
                if report_epoch != best_metric_epoch:
                    self.checkpoint_save(
                        ckpt=intermediate_ckpt_path, model=model, epoch=report_epoch, best_metric=val_acc_mean
                    )
                else:
                    try:
                        shutil.copyfile(best_ckpt_path, intermediate_ckpt_path)  # if already saved once
                    except Exception as err:
                        logger.warning(f"error copying {best_ckpt_path} {intermediate_ckpt_path} {err}")
                        pass

            if lr_scheduler is not None:
                lr_scheduler.step()

            if self.global_rank == 0:
                # report time estimate
                time_remaining_estimate = train_time * (num_epochs - epoch)
                if val_loader is not None and len(val_loader) > 0:
                    if validation_time == 0:
                        validation_time = train_time
                    time_remaining_estimate += validation_time * len(val_schedule_list)

                logger.info(
                    f"Estimated remaining training time for the current model fold {config('fold')} is "
                    f"{time_remaining_estimate/3600:.2f} hr, "
                    f"running time {(time.time() - pre_loop_time)/3600:.2f} hr, "
                    f"est total time {(time.time() - pre_loop_time + time_remaining_estimate)/3600:.2f} hr \n"
                )

        # end of main epoch loop
        train_loader = val_loader = optimizer = None

        # optionally validate best checkpoint
        logger.info(f"Checking to run final testing {config('run_final_testing')}")
        if config("run_final_testing"):
            if distributed:
                dist.barrier()
            _ckpt_name = best_ckpt_path if os.path.exists(best_ckpt_path) else intermediate_ckpt_path
            if not os.path.exists(_ckpt_name):
                logger.info(f"Unable to validate final no checkpoints found {best_ckpt_path}, {intermediate_ckpt_path}")
            else:
                # self._props.pop("network", None)
                # self._set_props.pop("network", None)
                gc.collect()
                torch.cuda.empty_cache()
                best_metric = self.run_final_testing(
                    pretrained_ckpt_path=_ckpt_name,
                    progress_path=progress_path,
                    best_metric_epoch=best_metric_epoch,
                    pre_loop_time=pre_loop_time,
                )

                if (
                    self.global_rank == 0
                    and tb_writer is not None
                    and mlflow_is_imported
                    and mlflow.is_tracking_uri_set()
                ):
                    mlflow.log_param("acc_testing", val_acc_mean)
                    mlflow.log_metric("acc_testing", val_acc_mean)

        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

            if mlflow_is_imported and mlflow.is_tracking_uri_set():
                mlflow.end_run()

        logger.info(
            f"=== DONE: best_metric: {best_metric:.4f} at epoch: {best_metric_epoch} of {report_num_epochs}."
            f"Training time {(time.time() - pre_loop_time)/3600:.2f} hr."
        )
        return best_metric

    def run_final_testing(self, pretrained_ckpt_path, progress_path, best_metric_epoch, pre_loop_time):
        logger.info("Running final best model testing set!")

        # validate
        start_time = time.time()

        self._props.pop("network", None)
        self.parser["pretrained_ckpt_path"] = pretrained_ckpt_path
        self.parser["validate#evaluator#postprocessing"] = None  # not saving images

        val_acc_mean, val_loss, val_acc = self.validate(val_key="testing")
        validation_time = f"{time.time() - start_time:.2f}s"
        val_acc_mean = float(np.mean(val_acc))
        logger.info(f"Testing: loss: {val_loss:.4f} acc_avg: {val_acc_mean:.4f} acc {val_acc} time {validation_time}")

        if self.global_rank == 0 and progress_path is not None:
            self.save_progress_yaml(
                progress_path=progress_path,
                ckpt=pretrained_ckpt_path,
                best_avg_score_epoch=best_metric_epoch,
                best_avg_score=val_acc_mean,
                validation_time=validation_time,
                run_final_testing=True,
                time=f"{(time.time() - pre_loop_time) / 3600:.2f} hr",
            )
        return val_acc_mean

    def validate(self, validation_files=None, val_key=None, datalist=None):
        if self.config("pretrained_ckpt_name", None) is None and self.config("pretrained_ckpt_path", None) is None:
            self.parser["pretrained_ckpt_name"] = "model.pt"
            logger.info("Using default model.pt checkpoint for validation.")

        grouping = self.config("validate#grouping", False)  # whether to computer average per datalist
        if validation_files is None:
            validation_files = self.read_val_datalists("validate", datalist, val_key=val_key, merge=not grouping)
        if len(validation_files) == 0:
            logger.warning(f"No validation files found {datalist} {val_key}!")
            return 0, 0, 0
        if not grouping or not isinstance(validation_files[0], (list, tuple)):
            validation_files = [validation_files]
        logger.info(f"validation file groups {len(validation_files)} grouping {grouping}")
        val_acc_dict = {}

        amp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            self.config("amp_dtype")
        ]
        if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            amp_dtype = torch.float16
            logger.warning(
                "bfloat16 dtype is not support on your device, changing to float16, use  --amp_dtype=float16 to set manually"
            )

        for datalist_id, group_files in enumerate(validation_files):
            self.set_val_datalist(group_files)
            val_loader = self.val_loader

            start_time = time.time()
            val_loss, val_acc = self.val_epoch(
                model=self.network,
                val_loader=val_loader,
                sliding_inferrer=self.config("inferer#sliding_inferer"),
                loss_function=self.config("loss_function"),
                acc_function=self.config("key_metric"),
                rank=self.rank,
                global_rank=self.global_rank,
                use_amp=self.config("amp"),
                amp_dtype=amp_dtype,
                post_transforms=self.config("validate#evaluator#postprocessing"),
                channels_last=self.config("channels_last"),
                device=self.config("device"),
            )
            val_acc_mean = float(np.mean(val_acc))
            logger.info(
                f"Validation {datalist_id} complete, loss_avg: {val_loss:.4f} "
                f"acc_avg: {val_acc_mean:.4f} acc {val_acc} time {time.time() - start_time:.2f}s"
            )
            val_acc_dict[datalist_id] = val_acc_mean
        for k, v in val_acc_dict.items():
            logger.info(f"group: {k} => {v:.4f}")
        val_acc_mean = sum(val_acc_dict.values()) / len(val_acc_dict.values())
        logger.info(f"Testing group score average: {val_acc_mean:.4f}")
        return val_acc_mean, val_loss, val_acc

    def infer(self, infer_files=None, infer_key=None, datalist=None):
        if self.config("pretrained_ckpt_name", None) is None and self.config("pretrained_ckpt_path", None) is None:
            self.parser["pretrained_ckpt_name"] = "model.pt"
            logger.info("Using default model.pt checkpoint for inference.")

        if infer_files is None:
            infer_files = self.read_val_datalists("infer", datalist, val_key=infer_key, merge=True)
        if len(infer_files) == 0:
            logger.warning(f"no file to infer {datalist} {infer_key}.")
            return
        logger.info(f"inference files {len(infer_files)}")
        self.set_val_datalist(infer_files)
        val_loader = self.val_loader

        amp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            self.config("amp_dtype")
        ]
        if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            logger.warning(
                "bfloat16 dtype is not support on your device, changing to float16, use  --amp_dtype=float16 to set manually"
            )

        start_time = time.time()
        self.val_epoch(
            model=self.network,
            val_loader=val_loader,
            sliding_inferrer=self.config("inferer#sliding_inferer"),
            loss_function=None,
            acc_function=None,
            rank=self.rank,
            global_rank=self.global_rank,
            use_amp=self.config("amp"),
            amp_dtype=amp_dtype,
            post_transforms=self.config("infer#evaluator#postprocessing"),
            channels_last=self.config("channels_last"),
            device=self.config("device"),
        )
        logger.info(f"Inference complete time {time.time() - start_time:.2f}s")
        return

    @torch.no_grad()
    def val_epoch(
        self,
        model,
        val_loader,
        sliding_inferrer,
        loss_function=None,
        acc_function=None,
        epoch=0,
        rank=0,
        global_rank=0,
        num_epochs=0,
        use_amp=True,
        amp_dtype=torch.float16,
        post_transforms=None,
        channels_last=False,
        device=None,
    ):
        model.eval()
        distributed = dist.is_available() and dist.is_initialized()
        memory_format = torch.channels_last if channels_last else torch.preserve_format

        run_loss = CumulativeAverage()
        run_acc = CumulativeAverage()
        run_loss.append(torch.tensor(0, device=device), count=0)

        avg_loss = avg_acc = 0
        start_time = time.time()

        # In DDP, each replica has a subset of data, but if total data length is not evenly divisible by num_replicas,
        # then some replicas has 1 extra repeated item.
        # For proper validation with batch of 1, we only want to collect metrics for non-repeated items,
        # hence let's compute a proper subset length
        nonrepeated_data_length = len(val_loader.dataset)
        sampler = val_loader.sampler
        if distributed and isinstance(sampler, DistributedSampler) and not sampler.drop_last:
            nonrepeated_data_length = len(range(sampler.rank, len(sampler.dataset), sampler.num_replicas))

        for idx, batch_data in enumerate(val_loader):
            data = batch_data["image"].as_subclass(torch.Tensor).to(memory_format=memory_format, device=device)
            filename = batch_data["image"].meta[ImageMetaKey.FILENAME_OR_OBJ]
            batch_size = data.shape[0]
            loss = acc = None

            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits = sliding_inferrer(inputs=data, network=model)
            data = None

            # calc loss
            if loss_function is not None:
                target = batch_data["flow"].as_subclass(torch.Tensor).to(device=logits.device)
                loss = loss_function(logits, target)
                run_loss.append(loss.to(device=device), count=batch_size)
                target = None

            pred_mask_all = []

            for b_ind in range(logits.shape[0]):  # go over batch dim
                pred_mask, p = LogitsToLabels()(logits=logits[b_ind], filename=filename)
                pred_mask_all.append(pred_mask)

            if acc_function is not None:
                label = batch_data["label"].as_subclass(torch.Tensor)

                for b_ind in range(label.shape[0]):
                    acc = acc_function(pred_mask_all[b_ind], label[b_ind, 0].long())
                    acc = acc.detach().clone() if isinstance(acc, torch.Tensor) else torch.tensor(acc)

                    if idx < nonrepeated_data_length:
                        run_acc.append(acc.to(device=device), count=1)
                    else:
                        run_acc.append(torch.zeros_like(acc, device=device), count=0)
                label = None

            avg_loss = loss.cpu() if loss is not None else 0
            avg_acc = acc.cpu().numpy() if acc is not None else 0

            logger.info(
                f"Val {epoch}/{num_epochs} {idx}/{len(val_loader)} "
                f"loss: {avg_loss:.4f} acc {avg_acc}  time {time.time() - start_time:.2f}s"
            )

            if post_transforms:
                seg = torch.from_numpy(np.stack(pred_mask_all, axis=0).astype(np.int32)).unsqueeze(1)
                batch_data["seg"] = convert_to_dst_type(
                    seg, batch_data["image"], dtype=torch.int32, device=torch.device("cpu")
                )[0]
                for bd in decollate_batch(batch_data):
                    post_transforms(bd)  # (currently only to save output mask)

            start_time = time.time()

        label = target = data = batch_data = None

        if distributed:
            dist.barrier()

        avg_loss = run_loss.aggregate()
        avg_acc = run_acc.aggregate()

        if np.any(avg_acc < 0):
            dist.barrier()
            logger.warning(f"Avg accuracy is negative ({avg_acc}), something went wrong!!!!!")

        return avg_loss, avg_acc

    def train_epoch(
        self,
        model,
        train_loader,
        optimizer,
        loss_function,
        acc_function,
        grad_scaler,
        epoch,
        rank,
        global_rank=0,
        num_epochs=0,
        use_amp=True,
        amp_dtype=torch.float16,
        channels_last=False,
        device=None,
    ):
        model.train()
        memory_format = torch.channels_last if channels_last else torch.preserve_format

        run_loss = CumulativeAverage()

        start_time = time.time()
        avg_loss = avg_acc = 0
        for idx, batch_data in enumerate(train_loader):
            data = batch_data["image"].as_subclass(torch.Tensor).to(memory_format=memory_format, device=device)
            target = batch_data["flow"].as_subclass(torch.Tensor).to(memory_format=memory_format, device=device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(data)

            # print('logits', logits.shape, logits.dtype)
            loss = loss_function(logits.float(), target)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            batch_size = data.shape[0]

            run_loss.append(loss, count=batch_size)
            avg_loss = run_loss.aggregate()

            logger.info(
                f"Epoch {epoch}/{num_epochs} {idx}/{len(train_loader)} "
                f"loss: {avg_loss:.4f} time {time.time() - start_time:.2f}s "
            )
            start_time = time.time()

        optimizer.zero_grad(set_to_none=True)

        data = None
        target = None
        batch_data = None

        return avg_loss, avg_acc

    def save_history_csv(self, csv_path=None, header=None, **kwargs):
        if csv_path is not None:
            if header is not None:
                with open(csv_path, "a") as myfile:
                    wrtr = csv.writer(myfile, delimiter="\t")
                    wrtr.writerow(header)
            if len(kwargs):
                with open(csv_path, "a") as myfile:
                    wrtr = csv.writer(myfile, delimiter="\t")
                    wrtr.writerow(list(kwargs.values()))

    def save_progress_yaml(self, progress_path=None, ckpt=None, **report):
        if ckpt is not None:
            report["model"] = ckpt

        report["date"] = str(datetime.now())[:19]

        if progress_path is not None:
            yaml.add_representer(
                float, lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:float", f"{value:.4f}")
            )
            with open(progress_path, "a") as progress_file:
                yaml.dump([report], stream=progress_file, allow_unicode=True, default_flow_style=None, sort_keys=False)

        logger.info("Progress:" + ",".join(f" {k}: {v}" for k, v in report.items()))

    def checkpoint_save(self, ckpt: str, model: torch.nn.Module, **kwargs):
        # save checkpoint and config
        save_time = time.time()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        if self.config("compile", False):
            # remove key prefix of compiled models
            state_dict = OrderedDict(
                (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k, v) for k, v in state_dict.items()
            )

        torch.save({"state_dict": state_dict, "config": self.parser.config, **kwargs}, ckpt)

        save_time = time.time() - save_time
        logger.info(f"Saving checkpoint process: {ckpt}, {kwargs}, save_time {save_time:.2f}s")

        return save_time

    def checkpoint_load(self, ckpt: str, model: torch.nn.Module, **kwargs):
        # load checkpoint
        if not os.path.isfile(ckpt):
            logger.warning("Invalid checkpoint file: " + str(ckpt))
            return
        checkpoint = torch.load(ckpt, map_location="cpu")

        # if self.config("compile", False):
        #     checkpoint["state_dict"] = OrderedDict((k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k, v) for k, v in checkpoint["state_dict"].items())

        model.load_state_dict(checkpoint["state_dict"], strict=True)
        epoch = checkpoint.get("epoch", 0)
        best_metric = checkpoint.get("best_metric", 0)

        if self.config("continue", False):
            if "epoch" in checkpoint:
                self.parser["start_epoch"] = checkpoint["epoch"]
            if "best_metric" in checkpoint:
                self.parser["best_metric"] = checkpoint["best_metric"]

        logger.info(
            f"=> loaded checkpoint {ckpt} (epoch {epoch}) "
            f"(best_metric {best_metric}) setting start_epoch {self.config('start_epoch')}"
        )
        self.parser["start_epoch"] = int(self.config("start_epoch")) + 1
        return

    def schedule_validation_epochs(self, num_epochs, num_epochs_per_validation=None, fraction=0.16) -> list:
        """
        Schedule of epochs to validate (progressively more frequently)
            num_epochs - total number of epochs
            num_epochs_per_validation - if provided use a linear schedule with this step
            init_step
        """

        if num_epochs_per_validation is None:
            x = (np.sin(np.linspace(0, np.pi / 2, max(10, int(fraction * num_epochs)))) * num_epochs).astype(int)
            x = np.cumsum(np.sort(np.diff(np.unique(x)))[::-1])
            x[-1] = num_epochs
            x = x.tolist()
        else:
            if num_epochs_per_validation >= num_epochs:
                x = [num_epochs_per_validation]
            else:
                x = list(range(num_epochs_per_validation, num_epochs, num_epochs_per_validation))

        if len(x) == 0:
            x = [0]

        return x


def main(**kwargs) -> None:
    workflow = VistaCell(**kwargs)
    workflow.initialize()
    workflow.run()
    workflow.finalize()


if __name__ == "__main__":
    # to be able to run directly as python scripts/workflow.py --config_file=...
    # for debugging and development

    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    # from scripts import *

    fire, fire_is_imported = optional_import("fire")
    if fire_is_imported:
        fire.Fire(main)
    else:
        print("Missing package: fire")
