from typing import Callable, Sequence, Union, Dict, List


import torch
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.data import box_utils

from .utils import detach_to_numpy


class cocometric_ignite(Metric):
    def __init__(self, output_transform: Callable = lambda x: x, 
        target_box_key="box", target_label_key="label",
        device: Union[str, torch.device] = torch.device("cpu")):
        self.target_box_key = target_box_key
        self.target_label_key = target_label_key
        self.pred_score_key = target_label_key + "_scores"
        self.coco_metric = COCOMetric(classes=["nodule"], iou_list=[0.1], max_detection=[100])
        super(cocometric_ignite, self).__init__(output_transform=output_transform, device=device)

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
            pred_boxes=[val_data_i[self.target_box_key] for val_data_i in self.val_outputs_all],
            pred_classes=[
                val_data_i[self.target_label_key] for val_data_i in self.val_outputs_all
            ],
            pred_scores=[val_data_i[self.pred_score_key] for val_data_i in self.val_outputs_all],
            gt_boxes=[val_data_i[self.target_box_key] for val_data_i in self.val_targets_all],
            gt_classes=[
                val_data_i[self.target_label_key] for val_data_i in self.val_targets_all
            ],
        )
        val_epoch_metric_dict = self.coco_metric(results_metric)[0]
        val_epoch_metric = val_epoch_metric_dict.values()
        val_epoch_metric = sum(val_epoch_metric) / len(val_epoch_metric)
        return val_epoch_metric
