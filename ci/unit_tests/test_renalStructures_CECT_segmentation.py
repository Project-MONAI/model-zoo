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
from monai.bundle import ConfigWorkflow
from monai.data import NibabelWriter
from parameterized import parameterized
from utils import check_workflow

TEST_CASE_1 = [  # train
    {
        "bundle_root": "models/renalStructures_CECT_segmentation",
        "val_interval": 1,
        "run_id": "training",
        "train#trainer#max_epochs": 1,
        "train#dataset#cache_rate": 0.0,
        "train#dataloader#batch_size": 1,
        "validate#dataset#cache_rate": 0.0,
    }
]
TEST_CASE_2 = [{**TEST_CASE_1[0], "run_id": "evaluate"}]  # evaluate

TEST_CASE_3 = [{"bundle_root": "models/renalStructures_CECT_segmentation", "run_id": "inference"}]  # inference


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


class TestRenalStructuresCECT(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()

        dataset_size = 12
        input_shape = (96, 96, 96)
        writer = NibabelWriter(output_dtype=np.uint8)

        for s in range(dataset_size):
            os.makedirs(os.path.join(self.dataset_dir, f"case_{s:03d}"))
            test_image_artery = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            test_image_vein = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            test_image_excret = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            test_label = np.random.randint(6, size=input_shape, dtype="uint8")
            image_filename_artery = os.path.join(self.dataset_dir, f"case_{s:03d}", "12.nii.gz")
            image_filename_vein = os.path.join(self.dataset_dir, f"case_{s:03d}", "22-.nii.gz")
            image_filename_excret = os.path.join(self.dataset_dir, f"case_{s:03d}", "32-.nii.gz")
            label_filename = os.path.join(self.dataset_dir, f"case_{s:03d}", "merged.nii.gz")

            writer.set_data_array(test_image_artery, channel_dim=None)
            writer.set_metadata({"affine": np.eye(4), "original_affine": np.eye(4)})
            writer.write(image_filename_artery)
            writer.set_data_array(test_image_vein, channel_dim=None)
            writer.set_metadata({"affine": np.eye(4), "original_affine": np.eye(4)})
            writer.write(image_filename_vein)
            writer.set_data_array(test_image_excret, channel_dim=None)
            writer.set_metadata({"affine": np.eye(4), "original_affine": np.eye(4)})
            writer.write(image_filename_excret)
            writer.set_data_array(test_label, channel_dim=None)
            writer.set_metadata({"affine": np.eye(4), "original_affine": np.eye(4)})
            writer.write(label_filename)

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    @parameterized.expand([TEST_CASE_1])
    def test_train_config(self, override):
        override["dataset_dir"] = self.dataset_dir
        bundle_root = override["bundle_root"]
        # ensure customized library can be loaded in subprocess
        sys.path.append(bundle_root)

        trainer = ConfigWorkflow(
            workflow_type="train",
            config_file=os.path.join(bundle_root, "configs/train.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(trainer, check_properties=True)

    @parameterized.expand([TEST_CASE_2])
    def test_eval_config(self, override):
        override["dataset_dir"] = self.dataset_dir
        bundle_root = override["bundle_root"]
        # ensure customized library can be loaded in subprocess
        sys.path.append(bundle_root)

        train_file = os.path.join(bundle_root, "configs/train.json")
        eval_file = os.path.join(bundle_root, "configs/evaluate.json")

        validator = ConfigWorkflow(
            # override train.json, thus set the workflow to "train" rather than "eval"
            workflow_type="train",
            config_file=[train_file, eval_file],
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(validator, check_properties=True)

    @parameterized.expand([TEST_CASE_3])
    def test_infer_config(self, override):
        override["dataset_dir"] = self.dataset_dir
        bundle_root = override["bundle_root"]

        inferrer = ConfigWorkflow(
            workflow_type="infer",
            config_file=os.path.join(bundle_root, "configs/inference.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(inferrer, check_properties=True)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = test_order
    unittest.main(testLoader=loader)
