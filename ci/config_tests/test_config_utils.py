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

import nibabel as nib
import numpy as np
import torch
from enums import ContextKeys
from monai.bundle import ConfigWorkflow


def save_data_by_type(image, filename, image_type):
    if image_type == ContextKeys.NIBABEL:
        filename = f"{filename}.nii.gz"
        nib.save(nib.Nifti1Image(image, np.eye(4)), filename)
    # TODO: add more input types
    else:
        raise NotImplementedError(f"image_type {image_type} is not supported.")


def produce_dataset(test_context):
    dataset_dir = test_context.get(ContextKeys.DATASET_DIR)
    if not dataset_dir:
        raise ValueError(f"'test_context' does not contain necessary key: {ContextKeys.DATASET_DIR}.")
    dataset_size = test_context.get(ContextKeys.DATASET_SIZE, 1)
    input_shape = test_context.get(ContextKeys.INPUT_SHAPE, (64, 64, 64))
    data_type = test_context.get(ContextKeys.DATA_TYPE, ContextKeys.NIBABEL)
    for s in range(dataset_size):
        test_image = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
        test_label = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
        image_filename = os.path.join(dataset_dir, f"image_{s}")
        label_filename = os.path.join(dataset_dir, f"label_{s}")
        save_data_by_type(test_image, image_filename, data_type)
        save_data_by_type(test_label, label_filename, data_type)

    print(f"produced fake dataset in {dataset_dir} for testing configs.")


def test_weights_consistency(weights_1_path, weights_2_path):
    m1 = torch.load(weights_1_path)
    m2 = torch.load(weights_2_path)
    result = True
    for k in m1.keys():
        if not torch.all(m1[k] == m2[k]):
            result = False

    return result


class TestBundleConfigs:
    """
    A standard test bundle config class
    """

    def __init__(self, bundle_root, test_context):
        self.bundle_root = bundle_root
        self.test_context = test_context
        self.inference_override = test_context.get(ContextKeys.INFER_OVERRIDE)

    def assign_override_default_info(self, override_dict):
        override_dict["bundle_root"] = self.bundle_root
        override_dict["dataset_dir"] = self.test_context.get(ContextKeys.DATASET_DIR)
        return override_dict

    def test_train_config(self):
        print("start testing train config:")
        train_filename = self.test_context.get(ContextKeys.TRAIN, "train.json")
        train_override = self.test_context.get(ContextKeys.TRAIN_OVERRIDE, {})
        train_override = self.assign_override_default_info(train_override)

        trainer = ConfigWorkflow(
            workflow="train",
            config_file=os.path.join(self.bundle_root, "configs", train_filename),
            logging_file=os.path.join(self.bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(self.bundle_root, "configs/metadata.json"),
            **train_override,
        )
        trainer.initialize()
        trainer.run()
        trainer.finalize()
        print(f"{train_filename} is tested correctly.")

    def test_train_determinism(self):
        print("start testing determinism:")
        # step 1: run train config with determinism train config override
        train_filename = self.test_context.get(ContextKeys.TRAIN, "train.json")
        train_override = self.test_context.get(ContextKeys.TRAIN_DETERMINISM_OVERRIDE, {})
        train_override = self.assign_override_default_info(train_override)

        trainer = ConfigWorkflow(
            workflow="train",
            config_file=os.path.join(self.bundle_root, "configs", train_filename),
            logging_file=os.path.join(self.bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(self.bundle_root, "configs/metadata.json"),
            **train_override,
        )
        trainer.initialize()
        trainer.run()
        trainer.finalize()
        # step 2: compare weights
        weights_1 = self.test_context.get(ContextKeys.DETERMINISM_WEIGHTS_1)
        weights_2 = self.test_context.get(ContextKeys.DETERMINISM_WEIGHTS_2)

        result = test_weights_consistency(
            os.path.join(self.bundle_root, "models", weights_1), os.path.join(self.bundle_root, "models", weights_2)
        )
        if result is False:
            raise ValueError("train config is non-deterministic.")
        print(f"{train_filename} is deterministic.")

    def test_inference_config(self):
        print("start testing inference config:")
        infer_filename = self.test_context.get(ContextKeys.INFER, "inference.json")
        infer_override = self.test_context.get(ContextKeys.INFER_OVERRIDE, {})
        infer_override = self.assign_override_default_info(infer_override)

        inferrer = ConfigWorkflow(
            workflow="inference",
            config_file=os.path.join(self.bundle_root, "configs", infer_filename),
            logging_file=os.path.join(self.bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(self.bundle_root, "configs/metadata.json"),
            **infer_override,
        )
        inferrer.initialize()
        inferrer.run()
        inferrer.finalize()
        print(f"{infer_filename} is tested correctly.")
