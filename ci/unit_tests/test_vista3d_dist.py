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

TEST_CASE_TRAIN_MGPU = [{"bundle_root": "models/vista3d", "patch_size": [32, 32, 32], "epochs": 2, "val_interval": 1}]

TEST_CASE_EVAL_MGPU = [{"bundle_root": "models/vista3d", "patch_size": [32, 32, 32]}]

TEST_CASE_TRAIN_CONTINUAL = [
    {"bundle_root": "models/vista3d", "patch_size": [32, 32, 32], "epochs": 2, "val_interval": 1, "finetune": False}
]


def test_order(test_name1, test_name2):
    def get_order(name):
        if "train_mgpu" in name:
            return 1
        if "train_continual" in name:
            return 2
        if "eval" in name:
            return 3
        return 4

    return get_order(test_name1) - get_order(test_name2)


class TestVista3d(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        self.dataset_size = 5
        input_shape = (64, 64, 64)
        for s in range(self.dataset_size):
            test_image = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            test_label = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            image_filename = os.path.join(self.dataset_dir, f"image_{s}.nii.gz")
            label_filename = os.path.join(self.dataset_dir, f"label_{s}.nii.gz")
            nib.save(nib.Nifti1Image(test_image, np.eye(4)), image_filename)
            nib.save(nib.Nifti1Image(test_label, np.eye(4)), label_filename)

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    @parameterized.expand([TEST_CASE_TRAIN_MGPU])
    def test_train_mgpu_config(self, override):
        train_size = self.dataset_size // 2
        train_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
            }
            for i in range(train_size)
        ]
        val_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
            }
            for i in range(train_size, self.dataset_size)
        ]
        override["train_datalist"] = train_datalist
        override["val_datalist"] = val_datalist

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

    @parameterized.expand([TEST_CASE_EVAL_MGPU])
    def test_eval_mgpu_config(self, override):
        train_size = self.dataset_size // 2
        train_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
            }
            for i in range(train_size)
        ]
        val_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
            }
            for i in range(train_size, self.dataset_size)
        ]
        override["train_datalist"] = train_datalist
        override["val_datalist"] = val_datalist

        bundle_root = override["bundle_root"]
        sys.path = [bundle_root] + sys.path
        config_files = [
            os.path.join(bundle_root, "configs/train.json"),
            os.path.join(bundle_root, "configs/train_continual.json"),
            os.path.join(bundle_root, "configs/evaluate.json"),
            os.path.join(bundle_root, "configs/mgpu_evaluate.json"),
        ]
        output_path = os.path.join(bundle_root, "configs/evaluate_override.json")
        n_gpu = torch.cuda.device_count()
        export_config_and_run_mgpu_cmd(
            config_file=config_files,
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            override_dict=override,
            output_path=output_path,
            ngpu=n_gpu,
        )

    @parameterized.expand([TEST_CASE_TRAIN_CONTINUAL])
    def test_train_continual_config(self, override):
        train_size = self.dataset_size // 2
        train_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
            }
            for i in range(train_size)
        ]
        val_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
            }
            for i in range(train_size, self.dataset_size)
        ]
        override["train_datalist"] = train_datalist
        override["val_datalist"] = val_datalist

        bundle_root = override["bundle_root"]
        sys.path = [bundle_root] + sys.path
        config_files = [
            os.path.join(bundle_root, "configs/train.json"),
            os.path.join(bundle_root, "configs/train_continual.json"),
            os.path.join(bundle_root, "configs/multi_gpu_train.json"),
        ]
        output_path = os.path.join(bundle_root, "configs/train_continual_override.json")
        n_gpu = torch.cuda.device_count()
        export_config_and_run_mgpu_cmd(
            config_file=config_files,
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            override_dict=override,
            output_path=output_path,
            ngpu=n_gpu,
        )


if __name__ == "__main__":
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = test_order
    unittest.main(testLoader=loader)
