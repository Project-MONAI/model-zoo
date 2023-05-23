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
from monai.bundle.scripts import run
from parameterized import parameterized

TEST_CASE = [
    {
        # override info
        "TRAIN_OVERRIDE": {
            "bundle_root": "models/spleen_ct_segmentation",
            "images": "$list(sorted(glob.glob(@dataset_dir + '/image_*.nii.gz')))",
            "labels": "$list(sorted(glob.glob(@dataset_dir + '/label_*.nii.gz')))",
            "epochs": 1,
            "train#dataset#cache_rate": 0.0,
            "validate#dataset#cache_rate": 0.0,
            "train#dataloader#num_workers": 1,
            "validate#dataloader#num_workers": 1,
            "train#random_transforms#0#spatial_size": [32, 32, 32],
        },
        "INFER_OVERRIDE": {
            "bundle_root": "models/spleen_ct_segmentation",
            "datalist": "$list(sorted(glob.glob(@dataset_dir + '/image_*.nii.gz')))",
        },
    }
]


class TestSpleenCTSeg(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
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

    @parameterized.expand([TEST_CASE])
    def test_configs(self, test_case):
        train_override, infer_override = test_case["TRAIN_OVERRIDE"], test_case["INFER_OVERRIDE"]
        train_override["dataset_dir"] = infer_override["dataset_dir"] = self.dataset_dir
        bundle_root = train_override["bundle_root"]
        train_file = os.path.join(bundle_root, "configs/train.json")
        eval_file = os.path.join(bundle_root, "configs/evaluate.json")
        infer_file = os.path.join(bundle_root, "configs/inference.json")
        meta_file = os.path.join(bundle_root, "configs/metadata.json")
        logging_file = os.path.join(bundle_root, "configs/logging.conf")
        # test train
        run(config_file=train_file, meta_file=meta_file, logging_file=logging_file, **train_override)
        # test evaluate
        run(config_file=[train_file, eval_file], meta_file=meta_file, logging_file=logging_file, **train_override)
        # test inference
        run(config_file=infer_file, meta_file=meta_file, logging_file=logging_file, **infer_override)

    @parameterized.expand([TEST_CASE])
    def test_required_properties(self, test_case):
        


if __name__ == "__main__":
    unittest.main()
