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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.engines.evaluator import SupervisedEvaluator
from monai.engines.utils import IterationEvents, default_metric_cmp_fn
from monai.transforms import Transform
from monai.utils import ForwardMode, IgniteInfo, min_version, optional_import
from monai.utils.enums import CommonKeys as Keys
from torch.utils.data import DataLoader

from .detection_inferer import RetinaNetInferer

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")

__all__ = ["DetectionEvaluator"]


def detection_prepare_val_batch(
    batchdata: List[Dict[str, torch.Tensor]],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    **kwargs,
) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    """
    Default function to prepare the data for current iteration.
    Args `batchdata`, `device`, `non_blocking` refer to the ignite API:
    https://pytorch.org/ignite/v0.4.8/generated/ignite.engine.create_supervised_trainer.html.
    `kwargs` supports other args for `Tensor.to()` API.
    Returns:
        image, label(optional).
    """
    inputs = [
        batch_data_i["image"].to(device=device, non_blocking=non_blocking, **kwargs) for batch_data_i in batchdata
    ]

    if isinstance(batchdata[0].get(Keys.LABEL), torch.Tensor):
        targets = [
            dict(
                label=batch_data_i["label"].to(device=device, non_blocking=non_blocking, **kwargs),
                box=batch_data_i["box"].to(device=device, non_blocking=non_blocking, **kwargs),
            )
            for batch_data_i in batchdata
        ]
        return (inputs, targets)
    return inputs, None


class DetectionEvaluator(SupervisedEvaluator):
    """
    Supervised detection evaluation method with image and label, inherits from ``SupervisedEvaluator`` and ``Workflow``.
    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be Iterable, typically be torch.DataLoader.
        network: detector to evaluate in the evaluator, should be regular PyTorch `torch.nn.Module`.
        epoch_length: number of iterations for one epoch, default to `len(val_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse expected data (usually `image`, `label` and other network args)
            from `engine.state.batch` for every iteration, for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `engine.state.batch` as inputs, return data will be stored in `engine.state.output`.
            if not provided, use `self._iteration()` instead. for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        mode: model forward mode during evaluation, should be 'eval' or 'train',
            which maps to `model.eval()` or `model.train()`, default to 'eval'.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        to_kwargs: dict of other args for `prepare_batch` API when converting the input data, except for
            `device`, `non_blocking`.
        amp_kwargs: dict of the args for `torch.cuda.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast.
    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Iterable | DataLoader,
        network: RetinaNetDetector,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = detection_prepare_val_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: RetinaNetInferer | None = None,
        postprocessing: Transform | None = None,
        key_val_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Sequence | None = None,
        amp: bool = False,
        mode: ForwardMode | str = ForwardMode.EVAL,
        event_names: list[str | EventEnum] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            network=network,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            inferer=inferer,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )

    def _register_decollate(self):
        """
        Register the decollate operation for batch data, will execute after model forward and loss forward.
        """

        @self.on(IterationEvents.MODEL_COMPLETED)
        def _decollate_data(engine: Engine) -> None:
            output_list = []
            for i in range(len(engine.state.output[Keys.IMAGE])):
                output_list.append({})
                for k in engine.state.output.keys():
                    if engine.state.output[k] is not None:
                        output_list[i][k] = engine.state.output[k][i]
            engine.state.output = output_list
