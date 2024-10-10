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

from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import numpy as np
import torch
from monai.engines.evaluator import SupervisedEvaluator
from monai.engines.utils import IterationEvents, default_metric_cmp_fn, default_prepare_batch
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform, reset_ops_id
from monai.utils import ForwardMode, IgniteInfo, RankFilter, min_version, optional_import
from monai.utils.enums import CommonKeys as Keys
from torch.utils.data import DataLoader

rearrange, _ = optional_import("einops", name="rearrange")

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")

__all__ = ["Vista3dEvaluator"]


class Vista3dEvaluator(SupervisedEvaluator):
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
        amp_kwargs: dict of the args for `torch.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.amp.autocast.
    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Iterable | DataLoader,
        network: torch.nn.Module,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_val_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Sequence | None = None,
        amp: bool = False,
        mode: ForwardMode | str = ForwardMode.EVAL,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
        hyper_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            network=network,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
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

        self.network = network
        self.device = device
        self.inferer = SimpleInferer() if inferer is None else inferer
        self.hyper_kwargs = hyper_kwargs
        self.logger.addFilter(RankFilter())

    def transform_points(self, point, affine):
        """transform point to the coordinates of the transformed image
        point: numpy array [bs, N, 3]
        """
        bs, n = point.shape[:2]
        point = np.concatenate((point, np.ones((bs, n, 1))), axis=-1)
        point = rearrange(point, "b n d -> d (b n)")
        point = affine @ point
        point = rearrange(point, "d (b n)-> b n d", b=bs)[:, :, :3]
        return point

    def check_prompts_format(self, label_prompt, points, point_labels):
        """check the format of user prompts
        label_prompt: [1,2,3,4,...,B] List of tensors
        points: [[[x,y,z], [x,y,z], ...]] List of coordinates of a single object
        point_labels: [[1,1,0,...]] List of scalar that matches number of points
        """
        # check prompt is given
        if label_prompt is None and points is None:
            everything_labels = self.hyper_kwargs.get("everything_labels", None)
            if everything_labels is not None:
                label_prompt = [torch.tensor(_) for _ in everything_labels]
                return label_prompt, points, point_labels
            else:
                raise ValueError("Prompt must be given for inference.")
        # check label_prompt
        if label_prompt is not None:
            if isinstance(label_prompt, list):
                if not np.all([len(_) == 1 for _ in label_prompt]):
                    raise ValueError("Label prompt must be a list of single scalar, [1,2,3,4,...,].")
                if not np.all([(x < 255).item() for x in label_prompt]):
                    raise ValueError("Current bundle only supports label prompt smaller than 255.")
                if points is None:
                    supported_list = list({i + 1 for i in range(132)} - {16, 18, 129, 130, 131})
                    if not np.all([x in supported_list for x in label_prompt]):
                        raise ValueError("Undefined label prompt detected. Provide point prompts for zero-shot.")
            else:
                raise ValueError("Label prompt must be a list, [1,2,3,4,...,].")
        # check points
        if points is not None:
            if point_labels is None:
                raise ValueError("Point labels must be given if points are given.")
            if not np.all([len(_) == 3 for _ in points]):
                raise ValueError("Points must be three dimensional (x,y,z) in the shape of [[x,y,z],...,[x,y,z]].")
            if len(points) != len(point_labels):
                raise ValueError("Points must match point labels.")
            if not np.all([_ in [-1, 0, 1, 2, 3] for _ in point_labels]):
                raise ValueError("Point labels can only be -1,0,1 and 2,3 for special flags.")
        if label_prompt is not None and points is not None:
            if len(label_prompt) != 1:
                raise ValueError("Label prompt can only be a single object if provided with point prompts.")
        # check point_labels
        if point_labels is not None:
            if points is None:
                raise ValueError("Points must be given if point labels are given.")
        return label_prompt, points, point_labels

    def _iteration(self, engine: SupervisedEvaluator, batchdata: dict[str, torch.Tensor]) -> dict:
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.

        Args:
            engine: `SupervisedEvaluator` to execute operation for an iteration.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        label_set = engine.hyper_kwargs.get("label_set", None)
        # this validation label set should be consistent with 'labels.unique()', used to generate fg/bg points
        val_label_set = engine.hyper_kwargs.get("val_label_set", label_set)
        # If user provide prompts in the inference, input image must contain original affine.
        # the point coordinates are from the original_affine space, while image here is after preprocess transforms.
        if engine.hyper_kwargs["user_prompt"]:
            inputs, label_prompt, points, point_labels = (
                batchdata["image"],
                batchdata.get("label_prompt", None),
                batchdata.get("points", None),
                batchdata.get("point_labels", None),
            )
            labels = None
            label_prompt, points, point_labels = self.check_prompts_format(label_prompt, points, point_labels)
            inputs = inputs.to(engine.device)
            # For N foreground object, label_prompt is [1, N], but the batch number 1 needs to be removed. Convert to [N, 1]
            label_prompt = (
                torch.as_tensor([label_prompt]).to(inputs.device)[0].unsqueeze(-1) if label_prompt is not None else None
            )
            # For points, the size can only be [1, K, 3], where K is the number of points for this single foreground object.
            if points is not None:
                points = torch.as_tensor([points])
                points = self.transform_points(
                    points, np.linalg.inv(inputs.affine[0]) @ inputs.meta["original_affine"][0].numpy()
                )
                points = torch.from_numpy(points).to(inputs.device)
            point_labels = torch.as_tensor([point_labels]).to(inputs.device) if point_labels is not None else None

        # If validation with ground truth label available.
        else:
            inputs, labels = engine.prepare_batch(
                batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs
            )
            # create label prompt, this should be consistent with the label prompt used for training.
            if label_set is None:
                output_classes = engine.hyper_kwargs["output_classes"]
                label_set = np.arange(output_classes).tolist()
            label_prompt = torch.tensor(label_set).to(engine.state.device).unsqueeze(-1)
            # point prompt is generated withing vista3d, provide empty points
            points = torch.zeros(label_prompt.shape[0], 1, 3).to(inputs.device)
            point_labels = -1 + torch.zeros(label_prompt.shape[0], 1).to(inputs.device)
            # validation for either auto or point.
            if engine.hyper_kwargs.get("val_head", "auto") == "auto":
                # automatic only validation
                # remove val_label_set, vista3d will not sample points from gt labels.
                val_label_set = None
            else:
                # point only validation
                label_prompt = None

        # put iteration outputs into engine.state
        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: labels}
        # execute forward computation
        with engine.mode(engine.network):
            if engine.amp:
                with torch.amp.autocast("cuda", **engine.amp_kwargs):
                    engine.state.output[Keys.PRED] = engine.inferer(
                        inputs=inputs,
                        network=engine.network,
                        point_coords=points,
                        point_labels=point_labels,
                        class_vector=label_prompt,
                        labels=labels,
                        label_set=val_label_set,
                    )
            else:
                engine.state.output[Keys.PRED] = engine.inferer(
                    inputs=inputs,
                    network=engine.network,
                    point_coords=points,
                    point_labels=point_labels,
                    class_vector=label_prompt,
                    labels=labels,
                    label_set=val_label_set,
                )
        inputs = reset_ops_id(inputs)
        # Add dim 0 for decollate batch
        engine.state.output["label_prompt"] = label_prompt.unsqueeze(0) if label_prompt is not None else None
        engine.state.output["points"] = points.unsqueeze(0) if points is not None else None
        engine.state.output["point_labels"] = point_labels.unsqueeze(0) if point_labels is not None else None
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return engine.state.output
