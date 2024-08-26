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

import os
import shutil
import sys
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
from parameterized import parameterized
from utils import export_config_and_run_mgpu_cmd

TEST_CASE_TRAIN_MGPU = [{"bundle_root": "models/vista2d", "workflow_type": "train", "train#trainer#max_epochs": 2}]


class TestVista2d(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        self.dataset_size = 5
        input_shape = (256, 256)
        for s in range(self.dataset_size):
            test_image = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            test_label = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            image_filename = os.path.join(self.dataset_dir, f"image_{s}.png")
            label_filename = os.path.join(self.dataset_dir, f"label_{s}.png")
            plt.imsave(image_filename, test_image, cmap="gray")
            plt.imsave(label_filename, test_label, cmap="gray")

        self.bundle_root = "models/vista2d"
        sys.path = [self.bundle_root] + sys.path

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    @parameterized.expand([TEST_CASE_TRAIN_MGPU])
    def test_train_mgpu_config(self, override):
        override["train#dataset#data"] = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{s}.png"),
                "label": os.path.join(self.dataset_dir, f"label_{s}.png"),
            }
            for s in range(self.dataset_size)
        ]
        override["dataset#data"] = override["train#dataset#data"]

        output_path = os.path.join(self.bundle_root, "configs/train_override.json")
        n_gpu = torch.cuda.device_count()
        export_config_and_run_mgpu_cmd(
            config_file=os.path.join(self.bundle_root, "configs/hyper_parameters.yaml"),
            meta_file=os.path.join(self.bundle_root, "configs/metadata.json"),
            custom_workflow="scripts.workflow.VistaCell",
            override_dict=override,
            output_path=output_path,
            ngpu=n_gpu,
        )


if __name__ == "__main__":
    unittest.main()
