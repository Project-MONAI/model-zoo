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
import os
import warnings
from typing import TYPE_CHECKING, Callable, Optional

from monai.handlers.classification_saver import ClassificationSaver
from monai.utils import IgniteInfo, evenly_divisible_all_gather, min_version, optional_import, string_list_all_gather

from .utils import detach_to_numpy

idist, _ = optional_import("ignite", IgniteInfo.OPT_IMPORT_VERSION, min_version, "distributed")
Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


class DetectionSaver(ClassificationSaver):
    """
    Event handler triggered on completing every iteration to save the classification predictions as json file.
    If running in distributed data parallel, only saves json file in the specified rank.

    """

    def __init__(
        self,
        output_dir: str = "./",
        filename: str = "predictions.json",
        overwrite: bool = True,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        name: Optional[str] = None,
        save_rank: int = 0,
        pred_box_key: str = "box",
        pred_label_key: str = "label",
        pred_score_key: str = "label_scores",
    ) -> None:
        """
        Args:
            output_dir: if `saver=None`, output json file directory.
            filename: if `saver=None`, name of the saved json file name.
            overwrite: if `saver=None`, whether to overwriting existing file content, if True,
                will clear the file before saving. otherwise, will append new content to the file.
            batch_transform: a callable that is used to extract the `meta_data` dictionary of
                the input images from `ignite.engine.state.batch`. the purpose is to get the input
                filenames from the `meta_data` and store with classification results together.
                `engine.state` and `batch_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            output_transform: a callable that is used to extract the model prediction data from
                `ignite.engine.state.output`. the first dimension of its output will be treated as
                the batch dimension. each item in the batch will be saved individually.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            name: identifier of logging.logger to use, defaulting to `engine.logger`.
            save_rank: only the handler on specified rank will save to json file in multi-gpus validation,
                default to 0.
            pred_box_key: box key in the prediction dict.
            pred_label_key: classification label key in the prediction dict.
            pred_score_key: classification score key in the prediction dict.

        """
        super().__init__(
            output_dir=output_dir,
            filename=filename,
            overwrite=overwrite,
            batch_transform=batch_transform,
            output_transform=output_transform,
            name=name,
            save_rank=save_rank,
            saver=None,
        )
        self.pred_box_key = pred_box_key
        self.pred_label_key = pred_label_key
        self.pred_score_key = pred_score_key

    def _finalize(self, _engine: Engine) -> None:
        """
        All gather classification results from ranks and save to json file.

        Args:
            _engine: Ignite Engine, unused argument.
        """
        ws = idist.get_world_size()
        if self.save_rank >= ws:
            raise ValueError("target save rank is greater than the distributed group size.")

        # self._outputs is supposed to be a list of dict
        # self._outputs[i] should be have at least three keys: pred_box_key, pred_label_key, pred_score_key
        # self._filenames is supposed to be a list of str
        outputs = self._outputs
        filenames = self._filenames
        if ws > 1:
            outputs = evenly_divisible_all_gather(outputs, concat=False)
            filenames = string_list_all_gather(filenames)

        if len(filenames) != len(outputs):
            warnings.warn(f"filenames length: {len(filenames)} doesn't match outputs length: {len(outputs)}.")

        # save to json file only in the expected rank
        if idist.get_rank() == self.save_rank:
            results = [
                {
                    self.pred_box_key: detach_to_numpy(o[self.pred_box_key]).tolist(),
                    self.pred_label_key: detach_to_numpy(o[self.pred_label_key]).tolist(),
                    self.pred_score_key: detach_to_numpy(o[self.pred_score_key]).tolist(),
                    "image": f,
                }
                for o, f in zip(outputs, filenames)
            ]

            with open(os.path.join(self.output_dir, self.filename), "w") as outfile:
                json.dump(results, outfile, indent=4)
