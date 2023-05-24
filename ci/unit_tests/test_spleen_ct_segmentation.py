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

import nibabel as nib
import numpy as np
from monai.bundle import ConfigWorkflow
from parameterized import parameterized

TEST_CASE_1 = [  # train, evaluate
    {
        "bundle_root": "models/spleen_ct_segmentation",
        "images": "$list(sorted(glob.glob(@dataset_dir + '/image_*.nii.gz')))",
        "labels": "$list(sorted(glob.glob(@dataset_dir + '/label_*.nii.gz')))",
        "epochs": 1,
        "train#dataset#cache_rate": 0.0,
        "validate#dataset#cache_rate": 0.0,
        "train#dataloader#num_workers": 1,
        "validate#dataloader#num_workers": 1,
        "train#random_transforms#0#spatial_size": [32, 32, 32],
    }
]

TEST_CASE_2 = [  # inference
    {
        "bundle_root": "models/spleen_ct_segmentation",
        "datalist": "$list(sorted(glob.glob(@dataset_dir + '/image_*.nii.gz')))",
    }
]


class TestSpleenCTSeg(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        self.produce_fake_dataset()

    def produce_fake_dataset(self):
        dataset_size = 10
        input_shape = (64, 64, 64)
        for s in range(dataset_size):
            test_image = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            test_label = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            image_filename = os.path.join(self.dataset_dir, f"image_{s}.nii.gz")
            label_filename = os.path.join(self.dataset_dir, f"label_{s}.nii.gz")
            nib.save(nib.Nifti1Image(test_image, np.eye(4)), image_filename)
            nib.save(nib.Nifti1Image(test_label, np.eye(4)), label_filename)

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
        trainer.check_properties()
        trainer.run()
        trainer.finalize()

        validator = ConfigWorkflow(
            workflow="eval",
            config_file=[train_file, eval_file],
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        validator.initialize()
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
        inferrer.check_properties()
        inferrer.run()
        inferrer.finalize()


if __name__ == "__main__":
    unittest.main()
