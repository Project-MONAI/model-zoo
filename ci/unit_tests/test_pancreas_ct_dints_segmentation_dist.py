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
import torch
from parameterized import parameterized
from utils import export_config_and_run_mgpu_cmd, export_overrided_config

TEST_CASE_1 = [
    {
        "bundle_root": "models/pancreas_ct_dints_segmentation",
        "data_list_file_path": "models/pancreas_ct_dints_segmentation/configs/dataset_0.json",
        "num_epochs": 1,
        "num_epochs_per_validation": 1,
        "num_epochs_warmup": 0,
        "num_sw_batch_size": 2,
        "patch_size": [32, 32, 32],
        "patch_size_valid": [32, 32, 32],
    }
]

TEST_CASE_2 = [
    {
        "bundle_root": "models/pancreas_ct_dints_segmentation",
        "train#trainer#max_epochs": 1,
        "train#dataset#cache_rate": 0,
        "validate#dataset#cache_rate": 0,
        "validate#inferer#roi_size": [32, 32, 32],
        "train#random_transforms#0#spatial_size": [32, 32, 32],
        "val_interval": 1,
    }
]


def test_order(test_name1, test_name2):
    def get_order(name):
        if "search" in name:
            return 1
        return 2

    return get_order(test_name1) - get_order(test_name2)


def get_searched_arch(path):
    file_list = os.listdir(path)
    arch_name = None
    for f in file_list:
        if "search_code" in f:
            arch_name = f
    if arch_name is None:
        raise ValueError("Cannot find searched architectures file.")

    return arch_name


class TestDintsMGPU(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        dataset_size = 20
        input_shape = (64, 64, 64)
        images_folder = os.path.join(self.dataset_dir, "imagesTr")
        labels_folder = os.path.join(self.dataset_dir, "labelsTr")
        os.makedirs(images_folder)
        os.makedirs(labels_folder)
        for s in range(dataset_size):
            test_image = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            test_label = np.random.randint(low=0, high=3, size=input_shape).astype(np.int8)
            image_filename = os.path.join(images_folder, f"pancreas_{s}.nii.gz")
            label_filename = os.path.join(labels_folder, f"pancreas_{s}.nii.gz")
            nib.save(nib.Nifti1Image(test_image, np.eye(4)), image_filename)
            nib.save(nib.Nifti1Image(test_label, np.eye(4)), label_filename)

        prepare_datalist_file = "models/pancreas_ct_dints_segmentation/scripts/prepare_datalist.py"
        datalist_file = "models/pancreas_ct_dints_segmentation/configs/dataset_0.json"
        cmd = f"python {prepare_datalist_file} --path {self.dataset_dir} --output {datalist_file} --train_size 12"
        call_status = subprocess.run(cmd, shell=True)
        call_status.check_returncode()

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    @parameterized.expand([TEST_CASE_1])
    def test_search(self, override):
        override["data_file_base_dir"] = self.dataset_dir
        override["arch_ckpt_path"] = os.path.join(override["bundle_root"], "models")
        output_path = "models/pancreas_ct_dints_segmentation/configs/search_override.json"
        export_overrided_config("models/pancreas_ct_dints_segmentation/configs/search.yaml", override, output_path)
        cmd = f"torchrun --standalone --nnodes=1 --nproc_per_node=2 -m scripts.search run {output_path}"
        env = os.environ.copy()
        # ensure customized library can be loaded in subprocess
        env["PYTHONPATH"] = override.get("bundle_root", ".")
        subprocess.check_call(cmd, shell=True, env=env)

    @parameterized.expand([TEST_CASE_2])
    def test_train_mgpu_config(self, override):
        override["dataset_dir"] = self.dataset_dir
        bundle_root = override["bundle_root"]
        arch_name = get_searched_arch(os.path.join(bundle_root, "models"))
        override["arch_ckpt_path"] = os.path.join(bundle_root, "models", arch_name)
        train_file = os.path.join(bundle_root, "configs/train.yaml")
        mgpu_train_file = os.path.join(bundle_root, "configs/multi_gpu_train.yaml")
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
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = test_order
    unittest.main(testLoader=loader)
