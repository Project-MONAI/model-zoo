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
import tempfile
import unittest

import numpy as np
import torch
from monai.data import PILWriter
from parameterized import parameterized
from utils import export_config_and_run_mgpu_cmd

TEST_CASE_1 = [
    {
        "bundle_root": "models/endoscopic_tool_segmentation",
        "train#trainer#max_epochs": 1,
        "train#dataloader#num_workers": 1,
        "validate#dataloader#num_workers": 1,
        "train#deterministic_transforms#3#spatial_size": [32, 32],
    }
]

TEST_CASE_2 = [
    {
        "bundle_root": "models/endoscopic_tool_segmentation",
        "validate#dataloader#num_workers": 4,
        "train#deterministic_transforms#3#spatial_size": [32, 32],
    }
]


def test_order(test_name1, test_name2):
    def get_order(name):
        if "train" in name:
            return 1
        if "eval" in name:
            return 2
        if "infer" in name:
            return 3
        return 4

    return get_order(test_name1) - get_order(test_name2)


class TestEndoscopicSegMGPU(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        dataset_size = 10
        writer = PILWriter(np.uint8)
        shape = (3, 256, 256)
        for mode in ["train", "val"]:
            for sub_folder in ["inbody", "outbody"]:
                sample_dir = os.path.join(self.dataset_dir, f"{mode}/{sub_folder}")
                os.makedirs(sample_dir)
                for s in range(dataset_size):
                    image = np.random.randint(low=0, high=5, size=(3, *shape)).astype(np.int8)
                    image_filename = os.path.join(sample_dir, f"{sub_folder}_{s}.jpg")
                    writer.set_data_array(image, channel_dim=0)
                    writer.write(image_filename, verbose=True)
                    label = np.random.randint(low=0, high=5, size=shape).astype(np.int8)
                    label_filename = os.path.join(sample_dir, f"{sub_folder}_{s}_seg.jpg")
                    writer.set_data_array(label, channel_dim=None)
                    writer.write(label_filename, verbose=True)

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
        override["dataset_dir"] = self.dataset_dir
        bundle_root = override["bundle_root"]
        train_file = os.path.join(bundle_root, "configs/train.json")
        evaluate_file = os.path.join(bundle_root, "configs/evaluate.json")
        mgpu_evaluate_file = os.path.join(bundle_root, "configs/multi_gpu_evaluate.json")
        output_path = os.path.join(bundle_root, "configs/evaluate_override.json")
        n_gpu = torch.cuda.device_count()
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
