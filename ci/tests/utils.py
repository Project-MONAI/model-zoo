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

import numpy as np
import nibabel as nib
import os
import monai
from monai.bundle import ConfigWorkflow
import tempfile
import shutil

def produce_test_data(dataset_dir, dataset_size, input_shape):
    for s in range(dataset_size):
        test_image = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
        test_label = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
        image_filename = os.path.join(dataset_dir, f"image_{s}.nii.gz")
        label_filename = os.path.join(dataset_dir, f"label_{s}.nii.gz")
        nib.save(nib.Nifti1Image(test_image, np.eye(4)), image_filename)
        nib.save(nib.Nifti1Image(test_label, np.eye(4)), label_filename)

class TestBundleConfigs:
    '''
    A standard test bundle config class
    '''
    def __init__(
            self,
            test_data,
            dataset_size = 16,
            input_shape = (224, 224, 224),
            check_determinism = False,
        ):
        self.input_shape = input_shape
        self.dataset_size = dataset_size
        self.check_determinism = check_determinism
        self.dataset_dir = tempfile.mkdtemp()
        self.test_data = test_data
        monai.utils.set_determinism(seed=123)
        produce_test_data(self.dataset_dir, self.dataset_size, self.input_shape)

    def assign_override_default_info(self, override_dict):
        override_dict["bundle_root"] = self.bundle_root
        override_dict["dataset_dir"] = self.dataset_dir
        return override_dict

    def test_train_config(self):
        if "train" in self.test_data:
            train_override = self.test_data["train"]
        else:
            train_override = {}
        train_override = self.assign_override_default_info(train_override)
        
        trainer = ConfigWorkflow(
            workflow="train",
            config_file=os.path.join(self.bundle_root, "configs/train.json"),
            logging_file=os.path.join(self.bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(self.bundle_root, "configs/metadata.json"),
            **train_override,
        )
        trainer.initialize()
        trainer.run()
        trainer.finalize()

    def test_all_configs(self):
        self.test_train_config()
        shutil.rmtree(self.dataset_dir)