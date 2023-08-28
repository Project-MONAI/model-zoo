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
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from monai.data import create_test_image_2d
from monai.utils import optional_import, set_determinism
from parameterized import parameterized
from utils import export_config_and_run_mgpu_cmd

savemat, _ = optional_import("scipy.io", name="savemat")
Image, _ = optional_import("PIL.Image")
set_determinism(123)

TEST_CASE_1 = [  # mgpu train
    {
        "bundle_root": "./models/pathology_nuclick_annotation",
        "train#trainer#max_epochs": 2,
        "train#dataloader#batch_size": 1,
        "train#dataloader#num_workers": 1,
        "validate#dataloader#num_workers": 1,
        "validate#dataloader#batch_size": 1,
    }
]

TEST_CASE_2 = [  # mgpu evaluate
    {
        "bundle_root": "./models/pathology_nuclick_annotation",
        "train#trainer#max_epochs": 2,
        "train#dataloader#batch_size": 1,
    }
]


def test_order(test_name1, test_name2):
    def get_order(name):
        if "train" in name:
            return 1
        if "eval" in name:
            return 2
        return 3

    return get_order(test_name1) - get_order(test_name2)


class TestNuclickAnnotationMGPU(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        self.output = os.path.join(self.dataset_dir, "CoNSePNuclei")

        dataset_size = 10
        shape = (1000, 1000)
        for sub_folder in ["Train", "Test"]:
            sample_dir = os.path.join(self.dataset_dir, sub_folder)
            image_dir = os.path.join(sample_dir, "Images")
            label_dir = os.path.join(sample_dir, "Labels")
            os.makedirs(image_dir)
            os.makedirs(label_dir)
            for s in range(dataset_size):
                img, _ = create_test_image_2d(shape[0], shape[1], 600)
                im = Image.fromarray(img * 255).convert("RGB")
                image_filename = os.path.join(image_dir, f"{sub_folder}_{s}.png")
                im.save(image_filename, "PNG")

                inst_type = np.random.randint(low=0, high=5, size=(500, 1)).astype(np.int8)
                inst_centroid = np.random.randint(low=0, high=1000, size=(500, 2)).astype(np.int8)
                label = {
                    "inst_map": (img * 255).astype(np.int8),
                    "type_map": (img * 255).astype(np.int8),
                    "inst_type": inst_type,
                    "inst_centroid": inst_centroid,
                }
                label_filename = os.path.join(label_dir, f"{sub_folder}_{s}.mat")
                savemat(label_filename, label)

        prepare_datalist_file = "models/pathology_nuclick_annotation/scripts/data_process.py"
        cmd = f"python {prepare_datalist_file} --input {self.dataset_dir} --output {self.output}"
        call_status = subprocess.run(cmd, shell=True)
        call_status.check_returncode()

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    @parameterized.expand([TEST_CASE_1])
    def test_train_mgpu_config(self, override):
        override["dataset_dir"] = self.output
        bundle_root = override["bundle_root"]
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
            check_config=True,
        )

    @parameterized.expand([TEST_CASE_2])
    def test_evaluate_mgpu_config(self, override):
        override["dataset_dir"] = self.output
        bundle_root = override["bundle_root"]
        train_file = os.path.join(bundle_root, "configs/train.json")
        evaluate_file = os.path.join(bundle_root, "configs/evaluate.json")
        mgpu_evaluate_file = os.path.join(bundle_root, "configs/multi_gpu_evaluate.json")
        output_path = os.path.join(bundle_root, "configs/evaluate_override.json")
        n_gpu = torch.cuda.device_count()
        sys.path.append(bundle_root)
        export_config_and_run_mgpu_cmd(
            config_file=[train_file, evaluate_file, mgpu_evaluate_file],
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            override_dict=override,
            output_path=output_path,
            ngpu=n_gpu,
            check_config=True,
        )


if __name__ == "__main__":
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = test_order
    unittest.main(testLoader=loader)
