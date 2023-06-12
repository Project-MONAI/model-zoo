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
import tempfile
import unittest

import numpy as np
from monai.bundle import ConfigWorkflow
from monai.data import PILWriter
from parameterized import parameterized

TEST_CASE_1 = [  # train, evaluate
    {
        "bundle_root": "models/endoscopic_inbody_classification",
        "train#trainer#max_epochs": 2,
        "train#dataloader#num_workers": 1,
        "validate#dataloader#num_workers": 1,
    }
]

TEST_CASE_2 = [{"bundle_root": "models/endoscopic_inbody_classification", "handlers#0#_disabled_": True}]  # inference


class TestEndoscopicCls(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        dataset_size = 10
        writer = PILWriter(np.uint8)
        shape = (3, 256, 256)
        for sub_folder in ["inbody", "outbody"]:
            sample_dir = os.path.join(self.dataset_dir, sub_folder)
            os.makedirs(sample_dir)
            for s in range(dataset_size):
                image = np.random.randint(low=0, high=5, size=shape).astype(np.int8)
                image_filename = os.path.join(sample_dir, f"{sub_folder}_{s}.jpg")
                writer.set_data_array(image, channel_dim=0)
                writer.write(image_filename, verbose=True)

        prepare_datalist_file = "models/endoscopic_inbody_classification/scripts/data_process.py"
        outpath = "models/endoscopic_inbody_classification/label"
        cmd = f"python {prepare_datalist_file} --datapath {self.dataset_dir} --outpath {outpath}"
        call_status = subprocess.run(cmd, shell=True)
        call_status.check_returncode()

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    @parameterized.expand([TEST_CASE_1])
    def test_train_eval_config(self, override):
        override["dataset_dir"] = self.dataset_dir
        bundle_root = override["bundle_root"]
        train_file = os.path.join(bundle_root, "configs/train.json")
        eval_file = os.path.join(bundle_root, "configs/evaluate.json")

        trainer = ConfigWorkflow(
            workflow="train",
            config_file=train_file,
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        trainer.initialize()
        # check required and optional properties
        check_result = trainer.check_properties()
        if check_result is not None and len(check_result) > 0:
            raise ValueError(f"check properties for train config failed: {check_result}")
        trainer.run()
        trainer.finalize()

        validator = ConfigWorkflow(
            # override train.json, thus set the workflow to "train" rather than "eval"
            workflow="train",
            config_file=[train_file, eval_file],
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        validator.initialize()
        check_result = validator.check_properties()
        if check_result is not None and len(check_result) > 0:
            raise ValueError(f"check properties for overrided train config failed: {check_result}")
        validator.run()
        validator.finalize()

    @parameterized.expand([TEST_CASE_2])
    def test_infer_config(self, override):
        override["dataset_dir"] = self.dataset_dir
        bundle_root = override["bundle_root"]

        inferrer = ConfigWorkflow(
            workflow="infer",
            config_file=os.path.join(bundle_root, "configs/inference.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        inferrer.initialize()
        # check required and optional properties
        check_result = inferrer.check_properties()
        if check_result is not None and len(check_result) > 0:
            raise ValueError(f"check properties for inference config failed: {check_result}")
        inferrer.run()
        inferrer.finalize()


if __name__ == "__main__":
    unittest.main()
