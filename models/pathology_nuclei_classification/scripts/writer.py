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
from typing import Dict, Mapping, Optional

import numpy as np
from monai.config import NdarrayOrTensor, PathLike
from monai.data import ImageWriter

logger = logging.getLogger(__name__)


class ClassificationWriter(ImageWriter):
    def __init__(self, label_index_map: Optional[Dict[str, str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.label_index_map = (
            label_index_map
            if label_index_map
            else {"0": "Other", "1": "Inflammatory", "2": "Epithelial", "3": "Spindle-Shaped"}
        )

    def set_data_array(
        self,
        data_array: NdarrayOrTensor,
        channel_dim: Optional[int] = 0,
        squeeze_end_dims: bool = True,
        contiguous: bool = False,
        **kwargs,
    ):
        self.data_obj: np.ndarray = super().create_backend_obj(data_array)

    def set_metadata(self, meta_dict: Optional[Mapping] = None, resample: bool = True, **options):
        pass

    def write(self, filename: PathLike, verbose: bool = False, **kwargs):
        super().write(filename, verbose=verbose)
        result = []
        for idx, score in enumerate(self.data_obj):
            name = f"label_{idx}"
            name = self.label_index_map.get(str(idx)) if self.label_index_map else name
            if name:
                result.append({"idx": idx, "label": name, "score": float(score)})

        with open(filename, "w") as fp:
            json.dump(result, fp)
