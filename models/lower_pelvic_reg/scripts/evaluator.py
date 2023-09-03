from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, Tuple, Dict, Optional, Union

import torch
from monai.data import MetaObj, MetaTensor
from monai.engines import Evaluator
from torch.utils.data import DataLoader

from monai.config import IgniteInfo, KeysCollection
from monai.engines.utils import IterationEvents, default_metric_cmp_fn, default_prepare_batch
from monai.engines.workflow import Workflow
from monai.inferers import Inferer, SimpleInferer
from monai.networks.utils import eval_mode, train_mode
from monai.transforms import Transform
from monai.utils import ForwardMode, ensure_tuple, min_version, optional_import
from monai.utils.enums import CommonKeys as Keys
from monai.utils.module import look_up_option

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")


def prepare_reg_batch(
        batchdata: Tuple[Dict],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        **kwargs,
) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    """
    Default function to prepare the data for current iteration.

    The input `batchdata` is a pair of dictionaries both with keys "image" and "label".
    All returned tensors are moved to the given device using the given non-blocking argument before being returned.

    This function implements the expected API for a `prepare_batch` callable in Ignite:
    https://pytorch.org/ignite/v0.4.8/generated/ignite.engine.create_supervised_trainer.html

    Args:
        batchdata: a pair of dictionaries both with keys "image" and "label"
        device: device to move every returned tensor to
        non_blocking: equivalent argument for `Tensor.to`
        kwargs: further arguments for `Tensor.to`

    Returns:
        moving, fixed: a pair of dictionaries both with keys "image" and "label".
    """
    moving, fixed = batchdata
    for k in ["image", "label"]:
        moving[k].to(device=device, non_blocking=non_blocking, **kwargs)
        fixed[k].to(device=device, non_blocking=non_blocking, **kwargs)
    return moving, fixed


class RegistrationEvaluator(Evaluator):
    """
    Standard registration evaluation method with moving and fixed images and labels(optional),
    inherits from evaluator and Workflow.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be Iterable. Each batch input should be a pair of
        dictionaries both with keys "image" and "label".
        network: network to evaluate in the evaluator, should be regular PyTorch `torch.nn.Module`.
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
        network: torch.nn.Module,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = prepare_reg_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
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
        self.inferer = SimpleInferer() if inferer is None else inferer

    def _iteration(self, engine, batchdata):
        """
        callback function for the Supervised Registration Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - moving_image: image Tensor data for model input, already moved to device.
            - moving_label: label Tensor data corresponding to the image, already moved to device.
            - fixed_image: image Tensor data for model input, already moved to device.
            - fixed_label: label Tensor data corresponding to the image, already moved to device.
            - ddf: dense displacement field which registers the moving towards fixed.
            - warped_image: moving image warped by the predicted ddf
            - warped_label: moving label warped by the predicted ddf

        Args:
            engine: `SupervisedEvaluator` to execute operation for an iteration, should be a pair of dictionaries both
            with keys "image" and "label".
            batchdata: input data for this iteration.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        moving, fixed = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)

        # put iteration outputs into engine.state
        engine.state.output = {
            "moving_image": moving["image"],
            "moving_label": moving["label"],
            "fixed_image": fixed["image"],
            "fixed_label": fixed["label"],
            "moving_name": moving["name"],
            "fixed_name": fixed["name"]
        }

        # execute forward computation
        with engine.mode(engine.network):
            if engine.amp:
                with torch.cuda.amp.autocast(**engine.amp_kwargs):
                    engine.state.output.update(
                        engine.inferer((moving, fixed), engine.network)
                    )
            else:
                engine.state.output.update(
                    engine.inferer((moving, fixed), engine.network)
                )
        for k, v in engine.state.output.items():
            if isinstance(v, MetaTensor):
                engine.state.output[k] = torch.tensor(v.get_array())
        engine.state.batch = None
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        return engine.state.output
