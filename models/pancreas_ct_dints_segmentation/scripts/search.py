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

import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Sequence, Union

import monai
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from monai import transforms
from monai.bundle import ConfigParser
from monai.data import ThreadDataLoader, partition_dataset
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter


def run(config_file: Union[str, Sequence[str]]):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = ConfigParser()
    parser.read_config(config_file)

    arch_ckpt_path = parser["arch_ckpt_path"]
    amp = parser["amp"]
    data_file_base_dir = parser["data_file_base_dir"]
    data_list_file_path = parser["data_list_file_path"]
    determ = parser["determ"]
    learning_rate = parser["learning_rate"]
    learning_rate_arch = parser["learning_rate_arch"]
    learning_rate_milestones = np.array(parser["learning_rate_milestones"])
    num_images_per_batch = parser["num_images_per_batch"]
    num_epochs = parser["num_epochs"]  # around 20k iterations
    num_epochs_per_validation = parser["num_epochs_per_validation"]
    num_epochs_warmup = parser["num_epochs_warmup"]
    num_sw_batch_size = parser["num_sw_batch_size"]
    output_classes = parser["output_classes"]
    overlap_ratio = parser["overlap_ratio"]
    patch_size_valid = parser["patch_size_valid"]
    ram_cost_factor = parser["ram_cost_factor"]
    print("[info] GPU RAM cost factor:", ram_cost_factor)

    train_transforms = parser.get_parsed_content("transform_train")
    val_transforms = parser.get_parsed_content("transform_validation")

    # deterministic training
    if determ:
        set_determinism(seed=0)

    print("[info] number of GPUs:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        # initialize the distributed training process, every GPU runs in a process
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
    else:
        world_size = 1
    print("[info] world_size:", world_size)

    with open(data_list_file_path, "r") as f:
        json_data = json.load(f)

    list_train = json_data["training"]
    list_valid = json_data["validation"]

    # training data
    files = []
    for _i in range(len(list_train)):
        str_img = os.path.join(data_file_base_dir, list_train[_i]["image"])
        str_seg = os.path.join(data_file_base_dir, list_train[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    train_files = files

    random.shuffle(train_files)

    train_files_w = train_files[: len(train_files) // 2]
    if torch.cuda.device_count() > 1:
        train_files_w = partition_dataset(
            data=train_files_w, shuffle=True, num_partitions=world_size, even_divisible=True
        )[dist.get_rank()]
    print("train_files_w:", len(train_files_w))

    train_files_a = train_files[len(train_files) // 2 :]
    if torch.cuda.device_count() > 1:
        train_files_a = partition_dataset(
            data=train_files_a, shuffle=True, num_partitions=world_size, even_divisible=True
        )[dist.get_rank()]
    print("train_files_a:", len(train_files_a))

    # validation data
    files = []
    for _i in range(len(list_valid)):
        str_img = os.path.join(data_file_base_dir, list_valid[_i]["image"])
        str_seg = os.path.join(data_file_base_dir, list_valid[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    val_files = files

    if torch.cuda.device_count() > 1:
        val_files = partition_dataset(data=val_files, shuffle=False, num_partitions=world_size, even_divisible=False)[
            dist.get_rank()
        ]
    print("val_files:", len(val_files))

    # network architecture
    if torch.cuda.device_count() > 1:
        device = torch.device(f"cuda:{dist.get_rank()}")
    else:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    if torch.cuda.device_count() > 1:
        train_ds_a = monai.data.CacheDataset(
            data=train_files_a, transform=train_transforms, cache_rate=1.0, num_workers=8
        )
        train_ds_w = monai.data.CacheDataset(
            data=train_files_w, transform=train_transforms, cache_rate=1.0, num_workers=8
        )
        val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
    else:
        train_ds_a = monai.data.CacheDataset(
            data=train_files_a, transform=train_transforms, cache_rate=0.125, num_workers=8
        )
        train_ds_w = monai.data.CacheDataset(
            data=train_files_w, transform=train_transforms, cache_rate=0.125, num_workers=8
        )
        val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.125, num_workers=2)

    train_loader_a = ThreadDataLoader(train_ds_a, num_workers=6, batch_size=num_images_per_batch, shuffle=True)
    train_loader_w = ThreadDataLoader(train_ds_w, num_workers=6, batch_size=num_images_per_batch, shuffle=True)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1, shuffle=False)

    model = parser.get_parsed_content("network")
    dints_space = parser.get_parsed_content("dints_space")

    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    post_pred = transforms.Compose(
        [transforms.EnsureType(), transforms.AsDiscrete(argmax=True, to_onehot=output_classes)]
    )
    post_label = transforms.Compose([transforms.EnsureType(), transforms.AsDiscrete(to_onehot=output_classes)])

    # loss function
    loss_func = parser.get_parsed_content("loss")

    # optimizer
    optimizer = torch.optim.SGD(
        model.weight_parameters(), lr=learning_rate * world_size, momentum=0.9, weight_decay=0.00004
    )
    arch_optimizer_a = torch.optim.Adam(
        [dints_space.log_alpha_a], lr=learning_rate_arch * world_size, betas=(0.5, 0.999), weight_decay=0.0
    )
    arch_optimizer_c = torch.optim.Adam(
        [dints_space.log_alpha_c], lr=learning_rate_arch * world_size, betas=(0.5, 0.999), weight_decay=0.0
    )

    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    # amp
    if amp:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            print("[info] amp enabled")

    # start a typical PyTorch training
    val_interval = num_epochs_per_validation
    best_metric = -1
    best_metric_epoch = -1
    idx_iter = 0

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(arch_ckpt_path, "Events"))

        with open(os.path.join(arch_ckpt_path, "accuracy_history.csv"), "a") as f:
            f.write("epoch\tmetric\tloss\tlr\ttime\titer\n")

    dataloader_a_iterator = iter(train_loader_a)

    start_time = time.time()
    for epoch in range(num_epochs):
        decay = 0.5 ** np.sum(
            [(epoch - num_epochs_warmup) / (num_epochs - num_epochs_warmup) > learning_rate_milestones]
        )
        lr = learning_rate * decay * world_size
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            print("-" * 10)
            print(f"epoch {epoch + 1}/{num_epochs}")
            print("learning rate is set to {}".format(lr))

        model.train()
        epoch_loss = 0
        loss_torch = torch.zeros(2, dtype=torch.float, device=device)
        epoch_loss_arch = 0
        loss_torch_arch = torch.zeros(2, dtype=torch.float, device=device)
        step = 0

        for batch_data in train_loader_w:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            if world_size == 1:
                for _ in model.weight_parameters():
                    _.requires_grad = True
            else:
                for _ in model.module.weight_parameters():
                    _.requires_grad = True
            dints_space.log_alpha_a.requires_grad = False
            dints_space.log_alpha_c.requires_grad = False

            optimizer.zero_grad()

            if amp:
                with autocast():
                    outputs = model(inputs)
                    if output_classes == 2:
                        loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
                    else:
                        loss = loss_func(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                if output_classes == 2:
                    loss = loss_func(torch.flip(outputs, dims=[1]), 1 - labels)
                else:
                    loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            loss_torch[0] += loss.item()
            loss_torch[1] += 1.0
            epoch_len = len(train_loader_w)
            idx_iter += 1

            if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                print("[{0}] ".format(str(datetime.now())[:19]) + f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

            if epoch < num_epochs_warmup:
                continue

            try:
                sample_a = next(dataloader_a_iterator)
            except StopIteration:
                dataloader_a_iterator = iter(train_loader_a)
                sample_a = next(dataloader_a_iterator)
            inputs_search, labels_search = (sample_a["image"].to(device), sample_a["label"].to(device))
            if world_size == 1:
                for _ in model.weight_parameters():
                    _.requires_grad = False
            else:
                for _ in model.module.weight_parameters():
                    _.requires_grad = False
            dints_space.log_alpha_a.requires_grad = True
            dints_space.log_alpha_c.requires_grad = True

            # linear increase topology and RAM loss
            entropy_alpha_c = torch.tensor(0.0).to(device)
            entropy_alpha_a = torch.tensor(0.0).to(device)
            ram_cost_full = torch.tensor(0.0).to(device)
            ram_cost_usage = torch.tensor(0.0).to(device)
            ram_cost_loss = torch.tensor(0.0).to(device)
            topology_loss = torch.tensor(0.0).to(device)

            probs_a, arch_code_prob_a = dints_space.get_prob_a(child=True)
            entropy_alpha_a = -((probs_a) * torch.log(probs_a + 1e-5)).mean()
            entropy_alpha_c = -(
                F.softmax(dints_space.log_alpha_c, dim=-1) * F.log_softmax(dints_space.log_alpha_c, dim=-1)
            ).mean()
            topology_loss = dints_space.get_topology_entropy(probs_a)

            ram_cost_full = dints_space.get_ram_cost_usage(inputs.shape, full=True)
            ram_cost_usage = dints_space.get_ram_cost_usage(inputs.shape)
            ram_cost_loss = torch.abs(ram_cost_factor - ram_cost_usage / ram_cost_full)

            arch_optimizer_a.zero_grad()
            arch_optimizer_c.zero_grad()

            combination_weights = (epoch - num_epochs_warmup) / (num_epochs - num_epochs_warmup)

            if amp:
                with autocast():
                    outputs_search = model(inputs_search)
                    if output_classes == 2:
                        loss = loss_func(torch.flip(outputs_search, dims=[1]), 1 - labels_search)
                    else:
                        loss = loss_func(outputs_search, labels_search)

                    loss += combination_weights * (
                        (entropy_alpha_a + entropy_alpha_c) + ram_cost_loss + 0.001 * topology_loss
                    )

                scaler.scale(loss).backward()
                scaler.step(arch_optimizer_a)
                scaler.step(arch_optimizer_c)
                scaler.update()
            else:
                outputs_search = model(inputs_search)
                if output_classes == 2:
                    loss = loss_func(torch.flip(outputs_search, dims=[1]), 1 - labels_search)
                else:
                    loss = loss_func(outputs_search, labels_search)

                loss += 1.0 * (
                    combination_weights * (entropy_alpha_a + entropy_alpha_c) + ram_cost_loss + 0.001 * topology_loss
                )

                loss.backward()
                arch_optimizer_a.step()
                arch_optimizer_c.step()

            epoch_loss_arch += loss.item()
            loss_torch_arch[0] += loss.item()
            loss_torch_arch[1] += 1.0

            if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                print(
                    "[{0}] ".format(str(datetime.now())[:19])
                    + f"{step}/{epoch_len}, train_loss_arch: {loss.item():.4f}"
                )
                writer.add_scalar("train_loss_arch", loss.item(), epoch_len * epoch + step)

        # synchronizes all processes and reduce results
        if torch.cuda.device_count() > 1:
            dist.barrier()
            dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

        loss_torch = loss_torch.tolist()
        loss_torch_arch = loss_torch_arch.tolist()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            print(
                f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}, "
                f"best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
            )

            if epoch >= num_epochs_warmup:
                loss_torch_arch_epoch = loss_torch_arch[0] / loss_torch_arch[1]
                print(
                    f"epoch {epoch + 1} average arch loss: {loss_torch_arch_epoch:.4f}, "
                    f"best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
                )

        if (epoch + 1) % val_interval == 0 or (epoch + 1) == num_epochs:
            torch.cuda.empty_cache()
            model.eval()
            with torch.no_grad():
                metric = torch.zeros((output_classes - 1) * 2, dtype=torch.float, device=device)
                metric_sum = 0.0
                metric_count = 0
                metric_mat = []
                val_images = None
                val_labels = None
                val_outputs = None

                _index = 0
                for val_data in val_loader:
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    roi_size = patch_size_valid
                    sw_batch_size = num_sw_batch_size

                    if amp:
                        with torch.cuda.amp.autocast():
                            pred = sliding_window_inference(
                                val_images,
                                roi_size,
                                sw_batch_size,
                                lambda x: model(x),
                                mode="gaussian",
                                overlap=overlap_ratio,
                            )
                    else:
                        pred = sliding_window_inference(
                            val_images,
                            roi_size,
                            sw_batch_size,
                            lambda x: model(x),
                            mode="gaussian",
                            overlap=overlap_ratio,
                        )
                    val_outputs = pred

                    val_outputs = post_pred(val_outputs[0, ...])
                    val_outputs = val_outputs[None, ...]
                    val_labels = post_label(val_labels[0, ...])
                    val_labels = val_labels[None, ...]

                    value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=False)

                    print(_index + 1, "/", len(val_loader), value)

                    metric_count += len(value)
                    metric_sum += value.sum().item()
                    metric_vals = value.cpu().numpy()
                    if len(metric_mat) == 0:
                        metric_mat = metric_vals
                    else:
                        metric_mat = np.concatenate((metric_mat, metric_vals), axis=0)

                    for _c in range(output_classes - 1):
                        val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                        val1 = 1.0 - torch.isnan(value[0, 0]).float()
                        metric[2 * _c] += val0 * val1
                        metric[2 * _c + 1] += val1

                    _index += 1

                # synchronizes all processes and reduce results
                if torch.cuda.device_count() > 1:
                    dist.barrier()
                    dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)

                metric = metric.tolist()
                if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                    for _c in range(output_classes - 1):
                        print("evaluation metric - class {0:d}:".format(_c + 1), metric[2 * _c] / metric[2 * _c + 1])
                    avg_metric = 0
                    for _c in range(output_classes - 1):
                        avg_metric += metric[2 * _c] / metric[2 * _c + 1]
                    avg_metric = avg_metric / float(output_classes - 1)
                    print("avg_metric", avg_metric)

                    if avg_metric > best_metric:
                        best_metric = avg_metric
                        best_metric_epoch = epoch + 1
                        best_metric_iterations = idx_iter

                    (node_a_d, arch_code_a_d, arch_code_c_d, arch_code_a_max_d) = dints_space.decode()
                    torch.save(
                        {
                            "node_a": node_a_d,
                            "arch_code_a": arch_code_a_d,
                            "arch_code_a_max": arch_code_a_max_d,
                            "arch_code_c": arch_code_c_d,
                            "iter_num": idx_iter,
                            "epochs": epoch + 1,
                            "best_dsc": best_metric,
                            "best_path": best_metric_iterations,
                        },
                        os.path.join(arch_ckpt_path, "search_code_" + str(idx_iter) + ".pt"),
                    )
                    print("saved new best metric model")

                    dict_file = {}
                    dict_file["best_avg_dice_score"] = float(best_metric)
                    dict_file["best_avg_dice_score_epoch"] = int(best_metric_epoch)
                    dict_file["best_avg_dice_score_iteration"] = int(idx_iter)
                    with open(os.path.join(arch_ckpt_path, "progress.yaml"), "w") as out_file:
                        _ = yaml.dump(dict_file, stream=out_file)

                    print(
                        "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            epoch + 1, avg_metric, best_metric, best_metric_epoch
                        )
                    )

                    current_time = time.time()
                    elapsed_time = (current_time - start_time) / 60.0
                    with open(os.path.join(arch_ckpt_path, "accuracy_history.csv"), "a") as f:
                        f.write(
                            "{0:d}\t{1:.5f}\t{2:.5f}\t{3:.5f}\t{4:.1f}\t{5:d}\n".format(
                                epoch + 1, avg_metric, loss_torch_epoch, lr, elapsed_time, idx_iter
                            )
                        )

                if torch.cuda.device_count() > 1:
                    dist.barrier()

            torch.cuda.empty_cache()

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        writer.close()

    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
