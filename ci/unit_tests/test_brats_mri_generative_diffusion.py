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
from parameterized import parameterized
from utils import check_workflow

TEST_CASE_1 = [
    {
        "bundle_root": "models/brats_mri_generative_diffusion",
        "images": "$list(sorted(glob.glob(@dataset_dir + '/image_*.nii.gz')))",
        "labels": "$list(sorted(glob.glob(@dataset_dir + '/label_*.nii.gz')))",
        "train#trainer#max_epochs": 1,
        "train#dataset#cache_rate": 0.0,
        "train_batch_size": 1,
        "train_patch_size": [64, 64, 64],
    }
]

TEST_CASE_2 = [
    {
        "bundle_root": "models/brats_mri_generative_diffusion",
        "images": "$list(sorted(glob.glob(@dataset_dir + '/image_*.nii.gz')))",
        "labels": "$list(sorted(glob.glob(@dataset_dir + '/label_*.nii.gz')))",
        "train#trainer#max_epochs": 1,
        "train#dataset#cache_rate": 0.0,
        "train_batch_size": 1,
        "train_patch_size": [64, 64, 64],
        "latent_shape": [8, 16, 16, 16],
    }
]

TEST_CASE_3 = [{"bundle_root": "models/brats_mri_generative_diffusion", "infer_patch_size": [64, 64, 64]}]


def test_order(test_name1, test_name2):
    # specify test order.
    # The "train_autoencoder.json" config should be tested first in order to
    # produce model weights for inference, and train diffusion.
    def get_order(name):
        if "autoencoder" in name:
            return 1 if "train" in name else 2
        if "diffusion" in name:
            return 3 if "train" in name else 4
        return 5

    return get_order(test_name1) - get_order(test_name2)


class TestLdm3d(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        dataset_size = 5
        input_shape = (256, 256, 128)
        sub_dir = os.path.join(self.dataset_dir, "Task01_BrainTumour")
        os.makedirs(sub_dir)
        data_list = []
        for s in range(dataset_size):
            test_image = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            test_label = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            image_filename = os.path.join(self.dataset_dir, f"image_{s}.nii.gz")
            label_filename = os.path.join(self.dataset_dir, f"label_{s}.nii.gz")
            nib.save(nib.Nifti1Image(test_image, np.eye(4)), image_filename)
            nib.save(nib.Nifti1Image(test_label, np.eye(4)), label_filename)
            sample_dict = {"image": image_filename, "label": label_filename}
            data_list.append(sample_dict)
        # prepare a datalist file that "monai.apps.DecathlonDataset" requires
        full_dict = {
            "name": "",
            "description": "",
            "reference": "",
            "licence": "",
            "tensorImageSize": "",
            "modality": "",
            "labels": "",
            "numTraining": 4,
            "numTest": 0,
            "training": data_list,
        }
        with open(os.path.join(sub_dir, "dataset.json"), "w") as f:
            json.dump(full_dict, f)

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    @parameterized.expand([TEST_CASE_1])
    def test_autoencoder_train(self, override):
        override["dataset_dir"] = self.dataset_dir
        bundle_root = override["bundle_root"]
        sys.path = [bundle_root] + sys.path

        trainer = ConfigWorkflow(
            workflow_type="train",
            config_file=os.path.join(bundle_root, "configs/train_autoencoder.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(trainer, check_properties=False)

    @parameterized.expand([TEST_CASE_3])
    def test_autoencoder_infer(self, override):
        override["dataset_dir"] = self.dataset_dir
        bundle_root = override["bundle_root"]
        sys.path = [bundle_root] + sys.path

        inferrer = ConfigWorkflow(
            workflow_type="infer",
            config_file=os.path.join(bundle_root, "configs/inference_autoencoder.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(inferrer, check_properties=False)

    @parameterized.expand([TEST_CASE_2])
    def test_diffusion_train(self, override):
        override["dataset_dir"] = self.dataset_dir
        bundle_root = override["bundle_root"]
        sys.path = [bundle_root] + sys.path
        autoencoder_file = os.path.join(bundle_root, "configs/train_autoencoder.json")
        diffusion_file = os.path.join(bundle_root, "configs/train_diffusion.json")

        trainer = ConfigWorkflow(
            workflow_type="train",
            config_file=[autoencoder_file, diffusion_file],
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(trainer, check_properties=False)

    @parameterized.expand([TEST_CASE_3])
    def test_diffusion_infer(self, override):
        override["dataset_dir"] = self.dataset_dir
        bundle_root = override["bundle_root"]
        sys.path = [bundle_root] + sys.path

        inferrer = ConfigWorkflow(
            workflow_type="infer",
            config_file=os.path.join(bundle_root, "configs/inference.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(inferrer, check_properties=False)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = test_order
    unittest.main(testLoader=loader)
