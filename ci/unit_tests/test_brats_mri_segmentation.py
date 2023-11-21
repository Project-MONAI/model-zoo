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

import nibabel as nib
import numpy as np
from monai.bundle import ConfigWorkflow
from parameterized import parameterized
from utils import check_workflow

TEST_CASE_1 = [  # train, evaluate
    {
        "bundle_root": "models/brats_mri_segmentation",
        "epochs": 1,
        "train#dataloader#num_workers": 1,
        "validate#dataloader#num_workers": 1,
        "train#random_transforms#0#roi_size": [32, 32, 32],
    }
]

TEST_CASE_2 = [  # inference
    {"bundle_root": "models/brats_mri_segmentation", "handlers#0#_disabled_": True, "inferer#roi_size": [64, 64, 64]}
]


class TestBratsSeg(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        dataset_size = 10
        shape = (64, 64, 64)
        for s in range(dataset_size):
            sample_dir = os.path.join(self.dataset_dir, "training", "HGG", f"sample_{s}")
            os.makedirs(sample_dir)
            for image_suffix in ["t1", "t2", "t1ce", "flair"]:
                test_image = np.random.randint(low=0, high=2, size=shape).astype(np.int8)
                image_filename = os.path.join(sample_dir, f"image_{s}_{image_suffix}.nii.gz")
                nib.save(nib.Nifti1Image(test_image, np.eye(4)), image_filename)

            test_label = np.random.randint(low=0, high=5, size=shape).astype(np.int8)
            label_filename = os.path.join(sample_dir, f"label_{s}_seg.nii.gz")
            nib.save(nib.Nifti1Image(test_label, np.eye(4)), label_filename)

        prepare_datalist_file = "models/brats_mri_segmentation/scripts/prepare_datalist.py"
        datalist_file = "models/brats_mri_segmentation/configs/datalist.json"
        cmd = f"python {prepare_datalist_file} --path {self.dataset_dir} --output {datalist_file} --train_size 6"
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
            workflow_type="train",
            config_file=train_file,
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(trainer, check_properties=True)

        validator = ConfigWorkflow(
            # override train.json, thus set the workflow to "train" rather than "eval"
            workflow_type="train",
            config_file=[train_file, eval_file],
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(validator, check_properties=True)

    @parameterized.expand([TEST_CASE_2])
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
    unittest.main()
