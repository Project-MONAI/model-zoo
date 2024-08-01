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

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized
from utils import export_config_and_run_mgpu_cmd

TEST_CASE_TRAIN_MGPU = [{"bundle_root": "models/maisi_ct_generative", "epochs": 2}]


class TestMAISI(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.dataset_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        shutil.rmtree(self.dataset_dir)

    def create_train_dataset(self):
        self.dataset_size = 5
        input_shape = (32, 32, 32, 4)
        mask_shape = (128, 128, 128)
        for s in range(self.dataset_size):
            test_image = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            test_label = np.random.randint(low=0, high=2, size=mask_shape).astype(np.int8)
            image_filename = os.path.join(self.dataset_dir, f"image_{s}.nii.gz")
            label_filename = os.path.join(self.dataset_dir, f"label_{s}.nii.gz")
            nib.save(nib.Nifti1Image(test_image, np.eye(4)), image_filename)
            nib.save(nib.Nifti1Image(test_label, np.eye(4)), label_filename)

    @parameterized.expand([TEST_CASE_TRAIN_MGPU])
    def test_train_mgpu_config(self, override):
        self.create_train_dataset()
        train_size = self.dataset_size // 2
        train_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
                "dim": [128, 128, 128],
                "spacing": [1.0, 1.0, 1.0],
                "top_region_index": [0, 1, 0, 0],
                "bottom_region_index": [0, 0, 0, 1],
                "fold": 0,
            }
            for i in range(train_size)
        ]
        override["train_datalist"] = train_datalist

        bundle_root = override["bundle_root"]
        sys.path = [bundle_root] + sys.path
        train_file = os.path.join(bundle_root, "configs/train.json")
        mgpu_train_file = os.path.join(bundle_root, "configs/multi_gpu_train.json")
        output_path = os.path.join(bundle_root, "configs/train_override.json")
        n_gpu = torch.cuda.device_count()
        export_config_and_run_mgpu_cmd(
            config_file=[train_file, mgpu_train_file],
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            override_dict=override,
            output_path=output_path,
            ngpu=n_gpu,
        )


if __name__ == "__main__":
    unittest.main()
