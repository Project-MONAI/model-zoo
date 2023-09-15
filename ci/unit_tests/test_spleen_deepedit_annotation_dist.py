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

import numpy as np
import torch
from monai.data import ITKWriter
from parameterized import parameterized
from utils import export_config_and_run_mgpu_cmd

TEST_CASE_1 = [  # mgpu train
    {
        "bundle_root": "models/spleen_deepedit_annotation",
        "images": "$list(sorted(glob.glob(@dataset_dir + '/image_*.nii.gz')))",
        "labels": "$list(sorted(glob.glob(@dataset_dir + '/label_*.nii.gz')))",
        "train#trainer#max_epochs": 1,
        "train#dataset#cache_rate": 0.0,
        "validate#dataset#cache_rate": 0.0,
        "spatial_size": [32, 32, 32],
    }
]


class TestDeepeditAnnoMGPU(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        dataset_size = 10
        input_shape = (64, 64, 64)
        writer = ITKWriter(output_dtype=np.uint8)
        for s in range(dataset_size):
            test_image = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            test_label = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            image_filename = os.path.join(self.dataset_dir, f"image_{s}.nii.gz")
            label_filename = os.path.join(self.dataset_dir, f"label_{s}.nii.gz")
            writer.set_data_array(test_image, channel_dim=None)
            writer.set_metadata({"affine": np.eye(4), "original_affine": -1 * np.eye(4)})
            writer.write(image_filename)
            writer.set_data_array(test_label, channel_dim=None)
            writer.set_metadata({"affine": np.eye(4), "original_affine": -1 * np.eye(4)})
            writer.write(label_filename)

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    @parameterized.expand([TEST_CASE_1])
    def test_train_mgpu_config(self, override):
        override["dataset_dir"] = self.dataset_dir
        bundle_root = override["bundle_root"]
        train_file = os.path.join(bundle_root, "configs/train.json")
        mgpu_train_file = os.path.join(bundle_root, "configs/multi_gpu_train.json")
        output_path = os.path.join(bundle_root, "configs/train_override.json")
        n_gpu = torch.cuda.device_count()
        sys.path = [bundle_root] + sys.path
        export_config_and_run_mgpu_cmd(
            config_file=[train_file, mgpu_train_file],
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            override_dict=override,
            output_path=output_path,
            ngpu=n_gpu,
            check_config=True,
        )


if __name__ == "__main__":
    unittest.main()
