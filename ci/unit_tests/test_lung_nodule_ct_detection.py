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

import json
import os
import shutil
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
from monai.bundle import ConfigWorkflow
from monai.data import create_test_image_3d
from monai.utils import set_determinism
from parameterized import parameterized
from utils import check_workflow

set_determinism(123)

TEST_CASE_1 = [  # train, evaluate
    {
        "bundle_root": "models/lung_nodule_ct_detection",
        "epochs": 3,
        "batch_size": 1,
        "val_interval": 2,
        "train#dataloader#num_workers": 1,
        "validate#dataloader#num_workers": 1,
    }
]

TEST_CASE_2 = [{"bundle_root": "models/lung_nodule_ct_detection"}]  # inference


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


class TestLungNoduleDetection(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        dataset_size = 3
        train_patch_size = (192, 192, 80)
        dataset_json = {}

        img, _ = create_test_image_3d(train_patch_size[0], train_patch_size[1], train_patch_size[2], 2)
        image_filename = os.path.join(self.dataset_dir, "image.nii.gz")
        nib.save(nib.Nifti1Image(img, np.eye(4)), image_filename)
        label = [0, 0]
        box = [[108, 119, 131, 142, 26, 37], [132, 147, 149, 164, 25, 40]]
        data = {"box": box, "image": image_filename, "label": label}
        dataset_json["training"] = [data for _ in range(dataset_size)]
        dataset_json["validation"] = [data for _ in range(dataset_size)]

        self.ds_file = os.path.join(self.dataset_dir, "dataset.json")
        with open(self.ds_file, "w") as fp:
            json.dump(dataset_json, fp, indent=2)

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    @parameterized.expand([TEST_CASE_1])
    def test_train_eval_config(self, override):
        override["dataset_dir"] = self.dataset_dir
        override["data_list_file_path"] = self.ds_file
        bundle_root = override["bundle_root"]
        train_file = os.path.join(bundle_root, "configs/train.json")
        eval_file = os.path.join(bundle_root, "configs/evaluate.json")

        sys.path.append(bundle_root)
        trainer = ConfigWorkflow(
            workflow_type="train",
            config_file=train_file,
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(trainer, check_properties=False)

        validator = ConfigWorkflow(
            # override train.json, thus set the workflow to "train" rather than "eval"
            workflow_type="train",
            config_file=[train_file, eval_file],
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(validator, check_properties=False)

    @parameterized.expand([TEST_CASE_2])
    def test_infer_config(self, override):
        override["dataset_dir"] = self.dataset_dir
        override["data_list_file_path"] = self.ds_file
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
