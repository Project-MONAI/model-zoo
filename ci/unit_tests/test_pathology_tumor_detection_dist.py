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

import csv
import os
import shutil
import tempfile
import unittest

import torch
from monai.apps.utils import download_url
from parameterized import parameterized
from utils import export_config_and_run_mgpu_cmd

TEST_CASE_1 = [
    {
        "bundle_root": "models/pathology_tumor_detection",
        "region_size": [256, 256],
        "num_epochs": 1,
        "epochs": 1,
        "train#dataloader#batch_size": 32,
        "validate#dataloader#batch_size": 32,
        "train#datalist#transform#func": "$lambda x: os.path.join(@dataset_dir, x + '.tiff')",
        "validate#datalist#transform#func": "$lambda x: os.path.join(@dataset_dir, x + '.tiff')",
    }
]

TIFF_INFO = {  # data to be downloaded for test
    "url": "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/CMU-1.tiff",
    "filename": "CMU-1.tiff",
    "hash_type": "sha256",
    "hash_val": "73a7e89bc15576587c3d68e55d9bf92f09690280166240b48ff4b48230b13bcd",
    "csv_info": ["CMU-1", 46000, 32914, 0, 0, 0, 0, 0, 0, 1, 1, 1],
}


class TestTumorDetectionMGPU(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        # download sample tiff file
        download_url(
            url=TIFF_INFO["url"],
            filepath=os.path.join(self.dataset_dir, TIFF_INFO["filename"]),
            hash_type=TIFF_INFO["hash_type"],
            hash_val=TIFF_INFO["hash_val"],
        )
        # prepare csv files
        for csv_file in ["training.csv", "validation.csv"]:
            with open(os.path.join("models/pathology_tumor_detection", csv_file), "w", newline="") as save_file:
                wr = csv.writer(save_file)
                wr.writerow(TIFF_INFO["csv_info"])
                # 2 gpu training need 2 samples
                wr.writerow(TIFF_INFO["csv_info"])

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
