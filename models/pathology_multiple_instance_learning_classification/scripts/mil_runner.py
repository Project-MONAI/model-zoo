import collections.abc
import csv
import logging
import os
import shutil
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from monai.config import KeysCollection
from monai.data import Dataset, load_decathlon_datalist
from monai.data.wsi_reader import WSIReader
from monai.metrics import Cumulative, CumulativeAverage
from monai.networks.nets import milmodel
from monai.transforms import (
    Compose,
    GridPatchd,
    LoadImaged,
    MapTransform,
    RandFlipd,
    RandGridPatchd,
    RandRotate90d,
    ScaleIntensityRanged,
    SplitDimd,
    ToTensord,
)
from monai.utils import require_pkg, set_determinism
from sklearn.metrics import cohen_kappa_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter


def train_epoch(model, loader, optimizer, scaler, epoch, rank, amp, epochs):
    """One train epoch over the dataset"""

    model.train()
    criterion = nn.BCEWithLogitsLoss()

    run_loss = CumulativeAverage()
    run_acc = CumulativeAverage()

    start_time = time.time()
    loss, acc = 0.0, 0.0

    for idx, batch_data in enumerate(loader):

        data = batch_data["image"].as_subclass(torch.Tensor).cuda(rank)
        target = batch_data["label"].as_subclass(torch.Tensor).cuda(rank)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            logits = model(data)
            loss = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = (logits.sigmoid().sum(1).detach().round() == target.sum(1).round()).float().mean()

        run_loss.append(loss)
        run_acc.append(acc)

        loss = run_loss.aggregate()
        acc = run_acc.aggregate()

        if rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, epochs, idx, len(loader)),
                "loss: {:.4f}".format(loss),
                "acc: {:.4f}".format(acc),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()

    optimizer.zero_grad(set_to_none=True)

    return loss, acc


@torch.no_grad()
def val_epoch(model, loader, epoch, distributed, rank, amp, epochs, calc_metric, max_tiles=None, output_dir=None):
    """One validation epoch over the dataset"""

    model.eval()

    model2 = model if not distributed else model.module
    has_extra_outputs = model2.mil_mode == "att_trans_pyramid"
    extra_outputs = model2.extra_outputs
    calc_head = model2.calc_head

    criterion = nn.BCEWithLogitsLoss()

    all_preds = Cumulative()
    all_targets = Cumulative()

    start_time = time.time()
    loss, acc = 0.0, 0.0
    run_loss = CumulativeAverage()
    run_acc = CumulativeAverage()

    for idx, batch_data in enumerate(loader):

        data = batch_data["image"].as_subclass(torch.Tensor).cuda(rank)

        with autocast(enabled=amp):

            if max_tiles is not None and data.shape[1] > max_tiles:
                # During validation, we want to use all instances/patches
                # and if its number is very big, we may run out of GPU memory
                # in this case, we first iteratively go over subsets of patches to calculate backbone features
                # and at the very end calculate the classification output

                logits = []
                logits2 = []

                for i in range(int(np.ceil(data.shape[1] / float(max_tiles)))):
                    data_slice = data[:, i * max_tiles : (i + 1) * max_tiles]
                    logits_slice = model(data_slice, no_head=True)
                    logits.append(logits_slice)

                    if has_extra_outputs:
                        logits2.append(
                            [
                                extra_outputs["layer1"],
                                extra_outputs["layer2"],
                                extra_outputs["layer3"],
                                extra_outputs["layer4"],
                            ]
                        )

                logits = torch.cat(logits, dim=1)
                if has_extra_outputs:
                    extra_outputs["layer1"] = torch.cat([l[0] for l in logits2], dim=0)
                    extra_outputs["layer2"] = torch.cat([l[1] for l in logits2], dim=0)
                    extra_outputs["layer3"] = torch.cat([l[2] for l in logits2], dim=0)
                    extra_outputs["layer4"] = torch.cat([l[3] for l in logits2], dim=0)

                logits = calc_head(logits)

            else:
                # if number of instances is not big, we can run inference directly
                logits = model(data)

        pred = logits.sigmoid().sum(1).detach().round()
        all_preds.extend(pred)

        if calc_metric:
            target = batch_data["label"].as_subclass(torch.Tensor).cuda(rank)
            loss = criterion(logits, target)

            target = target.sum(1).round()
            acc = (pred == target).float().mean()

            run_loss.append(loss)
            run_acc.append(acc)
            loss = run_loss.aggregate()
            acc = run_acc.get_current()

            all_targets.extend(target)

            if rank == 0:
                print(
                    "Val epoch {}/{} {}/{}".format(epoch, epochs, idx, len(loader)),
                    "loss: {:.4f}".format(loss),
                    "acc: {:.4f}".format(acc),
                    "time {:.2f}s".format(time.time() - start_time),
                )

            if output_dir is not None:
                for i in range(len(batch_data["filename"])):
                    case_name = batch_data["filename"][i]
                    acc = (pred[i] == target[i]).float().mean()
                    write_csv_row(
                        os.path.join(output_dir, "raw.csv"),
                        [case_name, int(pred[i].item()), int(target[i].item()), int(acc.item())],
                    )

        else:
            if rank == 0:
                print(
                    "Val epoch {}/{} {}/{}".format(epoch, epochs, idx, len(loader)),
                    "time {:.2f}s".format(time.time() - start_time),
                )
            if output_dir is not None:
                for i in range(len(batch_data["filename"])):
                    case_name = batch_data["filename"][i]
                    write_csv_row(os.path.join(output_dir, "raw.csv"), [case_name, int(pred[i].item())])

        start_time = time.time()

    if calc_metric:
        # Calculate QWK metric (Quadratic Weigted Kappa) https://en.wikipedia.org/wiki/Cohen%27s_kappa
        all_preds = all_preds.get_buffer().cpu().numpy().astype(np.float64)
        all_targets = all_targets.get_buffer().cpu().numpy().astype(np.float64)
        qwk = cohen_kappa_score(all_preds, all_targets, weights="quadratic")

        acc = np.mean((all_preds == all_targets).astype(np.float64))

        if output_dir is not None:
            write_csv_row(os.path.join(output_dir, "summary.csv"), ["avg_acc", "qwk"], mode="w")
            write_csv_row(os.path.join(output_dir, "summary.csv"), [acc, qwk])

    else:
        qwk = loss = acc = 0

    return loss, acc, qwk


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0):
    """Save checkpoint"""

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()  # remove module prefix if inside DDP
    else:
        state_dict = model.state_dict()

    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}

    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


class LabelEncodeIntegerGraded(MapTransform):
    """
    Convert an integer label to encoded array representation of length num_classes,
    with 1 filled in up to label index, and 0 otherwise. For example for num_classes=5,
    embedding of 2 -> (1,1,0,0,0)

    Args:
        num_classes: the number of classes to convert to encoded format.
        keys: keys of the corresponding items to be transformed. Defaults to ``'label'``.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(self, num_classes: int, keys: KeysCollection = "label", allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.num_classes = num_classes

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            label = int(d[key])
            lz = np.zeros(self.num_classes, dtype=np.float32)
            lz[:label] = 1.0
            d[key] = lz

        return d


def list_data_collate(batch: collections.abc.Sequence):
    """
    Combine instances from a list of dicts into a single dict, by stacking them along first dim
    [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
    followed by the default collate which will form a batch BxNx3xHxW
    """

    for i, item in enumerate(batch):
        data = item[0]
        data["image"] = torch.stack([ix["image"] for ix in item], dim=0)
        batch[i] = data
    return default_collate(batch)


def write_csv_row(filename, row, mode="a"):
    with open(filename, mode, encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


class MilRunner:
    def __init__(self, config):

        self.mil_config = config

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        if not os.path.exists(self.mil_config["ckpt_dir"]):
            os.makedirs(self.mil_config["ckpt_dir"], exist_ok=True)

        if self.mil_config["determ"]:
            set_determinism(seed=0)
        else:
            torch.backends.cudnn.benchmark = True

    def init_distributed(self):

        if dist.is_torchelastic_launched() and torch.cuda.device_count() > 1:
            dist.init_process_group(backend="nccl", init_method="env://")
            self.rank = int(os.getenv("LOCAL_RANK"))
            self.world_size = int(os.getenv("LOCAL_WORLD_SIZE"))
            torch.cuda.set_device(self.rank)
            self.distributed = dist.is_available() and dist.is_initialized()
        else:
            self.rank = 0
            self.world_size = 1
            self.distributed = False

    def get_datalist(self):

        datalist = self.mil_config.get("datalist", None)

        if datalist is None:
            # if datalist file is not provided, download the default json
            datalist = "./scripts/datalist_panda_0.json"
            if not os.path.exists(datalist) and self.rank == 0:

                @require_pkg(pkg_name="gdown")
                def download_datalist():
                    import gdown

                    resource_url = "https://drive.google.com/uc?id=1L6PtKBlHHyUgTE4rVhRuOLTQKgD4tBRK"
                    gdown.download(resource_url, datalist, quiet=False)

                download_datalist()

            if self.distributed:
                dist.barrier()

        return datalist

    def get_train_loader(self, data_list_key="training"):

        datalist = self.get_datalist()
        dataroot = self.mil_config["dataroot"]
        quick = self.mil_config["quick"]
        tile_size = self.mil_config["tile_size"]
        num_classes = self.mil_config["num_classes"]
        workers = self.mil_config["workers"]
        tile_count = self.mil_config["tile_count"]
        batch_size = self.mil_config["batch_size"]

        distributed = self.distributed

        training_list = load_decathlon_datalist(
            data_list_file_path=datalist, data_list_key=data_list_key, base_dir=dataroot
        )

        if quick:  # for debugging on a small subset
            training_list = training_list[:8]

        for item in training_list:
            item["filename"] = deepcopy(item["image"])

        train_transform = Compose(
            [
                LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=1, image_only=True),
                LabelEncodeIntegerGraded(keys=["label"], num_classes=num_classes),
                RandGridPatchd(
                    keys=["image"],
                    patch_size=(tile_size, tile_size),
                    num_patches=tile_count,
                    sort_fn="min",
                    pad_mode=None,
                    constant_values=255,
                ),
                SplitDimd(keys=["image"], dim=0, keepdim=False, list_output=True),
                RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
                RandRotate90d(keys=["image"], prob=0.5),
                ScaleIntensityRanged(keys=["image"], a_min=np.float32(255), a_max=np.float32(0)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        dataset_train = Dataset(data=training_list, transform=train_transform)
        train_sampler = DistributedSampler(dataset_train) if distributed else None

        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=workers,
            pin_memory=True,
            multiprocessing_context="spawn" if workers > 0 else None,
            sampler=train_sampler,
            collate_fn=list_data_collate,
        )

        return train_loader

    def get_val_loader(self, data_list_key="validation"):

        datalist = self.get_datalist()
        dataroot = self.mil_config["dataroot"]
        quick = self.mil_config["quick"]
        tile_size = self.mil_config["tile_size"]
        num_classes = self.mil_config["num_classes"]
        workers = self.mil_config["workers"]

        distributed = self.distributed

        validation_list = load_decathlon_datalist(
            data_list_file_path=datalist, data_list_key=data_list_key, base_dir=dataroot
        )

        for item in validation_list:
            item["filename"] = deepcopy(item["image"])

        if quick:  # for debugging on a small subset
            validation_list = validation_list[:8]

        valid_transform = Compose(
            [
                LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=1, image_only=True),
                LabelEncodeIntegerGraded(keys=["label"], num_classes=num_classes, allow_missing_keys=True),
                GridPatchd(
                    keys=["image"],
                    patch_size=(tile_size, tile_size),
                    threshold=0.999 * 3 * 255 * tile_size * tile_size,
                    pad_mode=None,
                    constant_values=255,
                ),
                SplitDimd(keys=["image"], dim=0, keepdim=False, list_output=True),
                ScaleIntensityRanged(keys=["image"], a_min=np.float32(255), a_max=np.float32(0)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        dataset_valid = Dataset(data=validation_list, transform=valid_transform)
        val_sampler = DistributedSampler(dataset_valid, shuffle=False) if distributed else None

        valid_loader = DataLoader(
            dataset_valid,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            multiprocessing_context="spawn" if workers > 0 else None,
            sampler=val_sampler,
            collate_fn=list_data_collate,
        )

        return valid_loader

    def get_model(self, pretrained_ckpt_path=None):

        distributed = self.distributed
        rank = self.rank

        num_classes = self.mil_config["num_classes"]
        mil_mode = self.mil_config["mil_mode"]

        pretrained_backbone = (pretrained_ckpt_path is None) or not os.path.isfile(
            pretrained_ckpt_path
        )  # if not loading a checkpoint, init from pretrained backbone

        model = milmodel.MILModel(num_classes=num_classes, pretrained=pretrained_backbone, mil_mode=mil_mode)

        best_acc = start_epoch = 0
        if not pretrained_backbone:
            checkpoint = torch.load(pretrained_ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
            if "best_acc" in checkpoint:
                best_acc = checkpoint["best_acc"]
            print(
                "=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(pretrained_ckpt_path, start_epoch, best_acc)
            )
        else:
            if self.rank == 0:
                print("Using pretrained backbone")

        model = model.cuda(rank)

        if distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

        return model

    def train(self):

        self.init_distributed()
        distributed = self.distributed
        rank = self.rank

        amp = self.mil_config["amp"]
        tile_count = self.mil_config["tile_count"]
        mil_mode = self.mil_config["mil_mode"]
        optim_lr = (
            self.mil_config["optim_lr"] * self.world_size
        ) / 2  # heuristic to scale up learning rate in multigpu setup
        epochs = self.mil_config["epochs"]
        weight_decay = self.mil_config["weight_decay"]
        val_every = self.mil_config["val_every"]

        ckpt_dir = self.mil_config["ckpt_dir"]

        writer = None
        if rank == 0:
            writer = SummaryWriter(log_dir=ckpt_dir)
            print("Writing Tensorboard logs to ", writer.log_dir)

        train_loader = self.get_train_loader()
        val_loader = self.get_val_loader()

        if rank == 0:
            print("Dataset training:", len(train_loader), "validation:", len(val_loader))

        model = self.get_model(pretrained_ckpt_path=self.mil_config["pretrained_ckpt_path"])
        params = model.parameters()

        if mil_mode in ["att_trans", "att_trans_pyramid"]:
            m = model if not distributed else model.module
            params = [
                {"params": list(m.attention.parameters()) + list(m.myfc.parameters()) + list(m.net.parameters())},
                {"params": list(m.transformer.parameters()), "lr": 6e-6, "weight_decay": 0.1},
            ]

        optimizer = torch.optim.AdamW(params, lr=optim_lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

        n_epochs = epochs
        val_acc_max = 0.0
        scaler = GradScaler(enabled=amp)

        # RUN TRAINING LOOP
        for epoch in range(n_epochs):

            if distributed and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
                dist.barrier()

            epoch_time = time.time()
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scaler=scaler, epoch=epoch, rank=rank, amp=amp, epochs=epochs
            )

            if rank == 0:
                print(
                    "Final training  {}/{}".format(epoch, n_epochs - 1),
                    "loss: {:.4f}".format(train_loss),
                    "acc: {:.4f}".format(train_acc),
                    "time {:.2f}s".format(time.time() - epoch_time),
                )

            if rank == 0 and writer is not None:
                writer.add_scalar("train_loss", train_loss, epoch)
                writer.add_scalar("train_acc", train_acc, epoch)

            b_new_best = False
            val_acc = 0
            if (epoch + 1) % val_every == 0:

                epoch_time = time.time()
                val_loss, val_acc, qwk = val_epoch(
                    model,
                    val_loader,
                    epoch=epoch,
                    distributed=distributed,
                    rank=rank,
                    amp=amp,
                    epochs=n_epochs,
                    calc_metric=True,
                    max_tiles=tile_count,
                )
                if rank == 0:
                    print(
                        "Final validation  {}/{}".format(epoch, n_epochs - 1),
                        "loss: {:.4f}".format(val_loss),
                        "acc: {:.4f}".format(val_acc),
                        "qwk: {:.4f}".format(qwk),
                        "time {:.2f}s".format(time.time() - epoch_time),
                    )
                    if writer is not None:
                        writer.add_scalar("val_loss", val_loss, epoch)
                        writer.add_scalar("val_acc", val_acc, epoch)
                        writer.add_scalar("val_qwk", qwk, epoch)

                    val_acc = qwk

                    if val_acc > val_acc_max:
                        print("qwk ({:.6f} --> {:.6f})".format(val_acc_max, val_acc))
                        val_acc_max = val_acc
                        b_new_best = True

            if rank == 0:
                save_checkpoint(model, epoch, best_acc=val_acc, filename=os.path.join(ckpt_dir, "model_final.pt"))
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(ckpt_dir, "model_final.pt"), os.path.join(ckpt_dir, "model.pt"))

            scheduler.step()

        if writer:
            writer.close()

        if distributed:
            dist.destroy_process_group()

        print("TRAINING DONE")

    def evaluate(self, data_list_key="validation", calc_metric=True):

        self.init_distributed()
        distributed = self.distributed
        rank = self.rank

        amp = self.mil_config["amp"]
        tile_count = self.mil_config["tile_count"]
        output_dir = self.mil_config["output_dir"]
        eval_ckpt_path = self.mil_config["eval_ckpt_path"]

        if eval_ckpt_path is None or not os.path.isfile(eval_ckpt_path):
            raise ValueError("eval_ckpt_path must be a valid file..." + str(eval_ckpt_path))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        header = ["case_name", "class", "target", "acc"] if calc_metric else ["case_name", "class"]
        write_csv_row(os.path.join(output_dir, "raw.csv"), header, mode="w")

        val_loader = self.get_val_loader(data_list_key=data_list_key)

        if rank == 0:
            print("Dataset len:", len(val_loader))

        model = self.get_model(pretrained_ckpt_path=eval_ckpt_path)

        epoch_time = time.time()
        val_loss, val_acc, qwk = val_epoch(
            model,
            val_loader,
            epoch=0,
            distributed=distributed,
            rank=rank,
            amp=amp,
            epochs=1,
            calc_metric=calc_metric,
            max_tiles=tile_count,
            output_dir=output_dir,
        )

        if rank == 0:
            if calc_metric:
                print(
                    "Final loss: {:.4f}".format(val_loss),
                    "acc: {:.4f}".format(val_acc),
                    "qwk: {:.4f}".format(qwk),
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
            else:
                print("Final loss: {:.4f}".format(val_loss), "time {:.2f}s".format(time.time() - epoch_time))
            print("ALL DONE")

        if self.distributed:
            dist.destroy_process_group()
