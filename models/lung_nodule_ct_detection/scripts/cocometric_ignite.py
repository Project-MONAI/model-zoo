from typing import Callable, Dict, Sequence, Union

import torch
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.data import box_utils

from .utils import detach_to_numpy


class IgniteCocoMetric(Metric):
    def __init__(
        self,
        coco_metric_monai: Union[None, COCOMetric] = None,
        box_key="box",
        label_key="label",
        pred_score_key="label_scores",
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device, None] = None,
        reduce_scalar: bool = True,
    ):
        r"""
        Computes coco detection metric in Ignite.

        Args:
            coco_metric_monai: the coco metric in monai.
                If not given, will asume COCOMetric(classes=[0], iou_list=[0.1], max_detection=[100])
            box_key: box key in the ground truth target dict and prediction dict.
            label_key: classification label key in the ground truth target dict and prediction dict.
            pred_score_key: classification score key in the prediction dict.
            output_transform: A callable that is used to transform the Engine’s
                process_function’s output into the form expected by the metric.
            device: specifies which device updates are accumulated on.
                Setting the metric’s device to be the same as your update arguments ensures
                the update method is non-blocking. By default, CPU.
            reduce_scalar: if True, will return the average value of coc metric values;
                if False, will return an dictionary of coc metric.

        Examples:
            To use with ``Engine`` and ``process_function``,
             simply attach the metric instance to the engine.
            The output of the engine's ``process_function`` needs to be in format of
            ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.
            For more information on how metric works with :class:`~ignite.engine.engine.Engine`,
             visit :ref:`attach-engine`.
            .. include:: defaults.rst
                :start-after: :orphan:
            .. testcode::
                coco = IgniteCocoMetric()
                coco.attach(default_evaluator, 'coco')
                preds = [
                    {
                        'box': torch.Tensor([[1,1,1,2,2,2]]),
                        'label':torch.Tensor([0]),
                        'label_scores':torch.Tensor([0.8])
                    }
                ]
                target = [{'box': torch.Tensor([[1,1,1,2,2,2]]), 'label':torch.Tensor([0])}]
                state = default_evaluator.run([[preds, target]])
                print(state.metrics['coco'])
            .. testoutput::
                1.0...
        .. versionadded:: 0.4.3
        """
        self.box_key = box_key
        self.label_key = label_key
        self.pred_score_key = pred_score_key
        if coco_metric_monai is None:
            self.coco_metric = COCOMetric(classes=[0], iou_list=[0.1], max_detection=[100])
        else:
            self.coco_metric = coco_metric_monai
        self.reduce_scalar = reduce_scalar

        if device is None:
            device = torch.device("cpu")
        super(IgniteCocoMetric, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.val_targets_all = []
        self.val_outputs_all = []

    @reinit__is_reduced
    def update(self, output: Sequence[Dict]) -> None:
        y_pred, y = output[0], output[1]
        self.val_outputs_all += y_pred
        self.val_targets_all += y

    @sync_all_reduce("val_targets_all", "val_outputs_all")
    def compute(self) -> float:
        self.val_outputs_all = detach_to_numpy(self.val_outputs_all)
        self.val_targets_all = detach_to_numpy(self.val_targets_all)

        results_metric = matching_batch(
            iou_fn=box_utils.box_iou,
            iou_thresholds=self.coco_metric.iou_thresholds,
            pred_boxes=[val_data_i[self.box_key] for val_data_i in self.val_outputs_all],
            pred_classes=[val_data_i[self.label_key] for val_data_i in self.val_outputs_all],
            pred_scores=[val_data_i[self.pred_score_key] for val_data_i in self.val_outputs_all],
            gt_boxes=[val_data_i[self.box_key] for val_data_i in self.val_targets_all],
            gt_classes=[val_data_i[self.label_key] for val_data_i in self.val_targets_all],
        )
        val_epoch_metric_dict = self.coco_metric(results_metric)[0]

        if self.reduce_scalar:
            val_epoch_metric = val_epoch_metric_dict.values()
            val_epoch_metric = sum(val_epoch_metric) / len(val_epoch_metric)
            return val_epoch_metric
        else:
            return val_epoch_metric_dict
