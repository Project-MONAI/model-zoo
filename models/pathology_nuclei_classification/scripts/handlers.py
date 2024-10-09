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

import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import numpy as np
import torch
import torch.distributed
from monai.utils import IgniteInfo, min_version, optional_import
from sklearn.metrics import classification_report

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
make_grid, _ = optional_import("torchvision.utils", name="make_grid")
Image, _ = optional_import("PIL.Image")
ImageDraw, _ = optional_import("PIL.ImageDraw")

if TYPE_CHECKING:
    from ignite.engine import Engine
    from torch.utils.tensorboard import SummaryWriter
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")


class TensorBoardImageHandler:
    def __init__(
        self,
        class_names,
        summary_writer: Optional[SummaryWriter] = None,
        log_dir: str = "./runs",
        tag_name="val",
        interval: int = 1,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        batch_limit=1,
        device=None,
    ) -> None:
        self.class_names = class_names
        self.writer = SummaryWriter(log_dir=log_dir) if summary_writer is None else summary_writer
        self.tag_name = tag_name
        self.interval = interval
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.batch_limit = batch_limit
        self.device = device

        self.logger = logging.getLogger(__name__)

        if torch.distributed.is_initialized():
            self.tag_name = f"{self.tag_name}-r{torch.distributed.get_rank()}"
        self.class_y: List[Any] = []
        self.class_y_pred: List[Any] = []

    def attach(self, engine: Engine) -> None:
        if self.interval == 1:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self, "iteration")
        engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self, "epoch")

    def __call__(self, engine: Engine, action) -> None:
        epoch = engine.state.epoch
        batch_data = self.batch_transform(engine.state.batch)
        output_data = self.output_transform(engine.state.output)

        if action == "iteration":
            for bidx in range(len(batch_data)):
                y = output_data[bidx]["label"].detach().cpu().numpy()
                y_pred = output_data[bidx]["pred"].detach().cpu().numpy()

                self.class_y.append(np.argmax(y))
                self.class_y_pred.append(np.argmax(y_pred))
            return

        self.write_metrics(epoch)
        self.write_images(batch_data, output_data, epoch)
        self.writer.flush()

    def write_images(self, batch_data, output_data, epoch):
        for bidx in range(len(batch_data)):
            image = batch_data[bidx]["image"].detach().cpu().numpy()
            y = output_data[bidx]["label"].detach().cpu().numpy()
            y_pred = output_data[bidx]["pred"].detach().cpu().numpy()

            sig_np = image[:3] * 128 + 128
            sig_np[0, :, :] = np.where(image[3] > 0, 1, sig_np[0, :, :])

            y_c = np.argmax(y)
            y_pred_c = np.argmax(y_pred)

            tag_prefix = f"{self.tag_name} - b{bidx} - " if self.batch_limit != 1 else f"{self.tag_name} - "
            label_pred_tag = f"{tag_prefix}Image/Signal/Label/Pred:"

            y_img = Image.new("RGB", image.shape[-2:])
            draw = ImageDraw.Draw(y_img)
            draw.text((10, 50), self.class_names.get(f"{y_c}", f"{y_c}"))

            y_pred_img = Image.new("RGB", image.shape[-2:], "green" if y_c == y_pred_c else "red")
            draw = ImageDraw.Draw(y_pred_img)
            draw.text((10, 50), self.class_names.get(f"{y_pred_c}", f"{y_pred_c}"))

            img_tensor = make_grid(
                tensor=[
                    torch.from_numpy(sig_np),
                    torch.from_numpy(np.stack((np.where(image[3] > 0, 255, 0),) * 3)),
                    torch.from_numpy(np.moveaxis(np.array(y_img), -1, 0)),
                    torch.from_numpy(np.moveaxis(np.array(y_pred_img), -1, 0)),
                ],
                nrow=4,
                normalize=True,
                pad_value=10,
            )
            self.writer.add_image(tag=label_pred_tag, img_tensor=img_tensor, global_step=epoch)

            if self.batch_limit == 1 or bidx == (self.batch_limit - 1):
                break

    def write_metrics(self, epoch):
        cr = classification_report(self.class_y, self.class_y_pred, output_dict=True, zero_division=0)
        for k, v in cr.items():
            if isinstance(v, dict):
                ltext = []
                cname = self.class_names.get(k, k)
                for n, m in v.items():
                    ltext.append(f"{n} => {m:.4f}")
                    self.writer.add_scalar(f"{self.tag_name}_cr_{cname}_{n}", m, epoch)

                self.logger.info(f"{self.tag_name} => Epoch[{epoch}] - {cname} - Metrics -- {'; '.join(ltext)}")
            else:
                self.logger.info(f"{self.tag_name} => Epoch[{epoch}] Metrics -- {k} => {v:.4f}")
                self.writer.add_scalar(f"{self.tag_name}_cr_{k}", v, epoch)

        self.class_y = []
        self.class_y_pred = []
