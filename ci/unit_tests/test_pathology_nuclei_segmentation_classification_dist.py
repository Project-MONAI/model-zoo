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
from parameterized import parameterized
from utils import export_config_and_run_mgpu_cmd

TEST_CASE_1 = [  # mgpu train
    {
        "bundle_root": "models/pathology_nuclei_segmentation_classification",
        "epochs": 1,
        "batch_size": 3,
        "train#dataloader#num_workers": 1,
        "validate#dataloader#num_workers": 1,
    }
]


class TestNucleiSegClsMGPU(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        dataset_size = 10
        # train
        for mode in ["Train", "Test"]:
            train_sample_dir = os.path.join(self.dataset_dir, mode)
            os.makedirs(train_sample_dir)
            for s in range(dataset_size):
                for image_suffix in ["image", "inst_map", "type_map"]:
                    if image_suffix == "image":
                        shape = (256, 256, 3)
                    else:
                        shape = (256, 256, 1)
                    test_image = np.random.randint(low=0, high=2, size=shape).astype(np.int8)
                    image_filename = os.path.join(train_sample_dir, f"{s}_{image_suffix}.npy")
                    np.save(image_filename, test_image)

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


if __name__ == "__main__":
    unittest.main()
