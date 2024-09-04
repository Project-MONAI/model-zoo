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

import nibabel as nib
import numpy as np
from monai.bundle import create_workflow
from monai.transforms import LoadImage
from parameterized import parameterized

TEST_CASE_INFER_1 = [
    {
        "bundle_root": "models/maisi_ct_generative",
        "num_output_samples": 1,
        "num_inference_steps": 2,
        "output_size": [256, 256, 256],
        "body_region": ["abdomen"],
        "anatomy_list": ["liver"],
    }
]

# This case will definitely trigger find_closest_masks func
TEST_CASE_INFER_2 = [
    {
        "bundle_root": "models/maisi_ct_generative",
        "num_output_samples": 1,
        "num_inference_steps": 2,
        "output_size": [256, 256, 128],
        "spacing": [1.5, 1.5, 1.5],
        "body_region": ["abdomen"],
        "anatomy_list": ["liver"],
    }
]

# This case will definitely trigger data augmentation for tumors
TEST_CASE_INFER_3 = [
    {
        "bundle_root": "models/maisi_ct_generative",
        "num_output_samples": 1,
        "num_inference_steps": 2,
        "output_size": [256, 256, 128],
        "spacing": [1.5, 1.5, 1.5],
        "body_region": ["abdomen"],
        "anatomy_list": ["bone lesion"],
    }
]

TEST_CASE_INFER_WITH_MASK_GENERATION = [
    {
        "bundle_root": "models/maisi_ct_generative",
        "num_output_samples": 1,
        "num_inference_steps": 2,
        "mask_generation_num_inference_steps": 2,
        "output_size": [256, 256, 256],
        "spacing": [1.5, 1.5, 2.0],
        "body_region": ["chest"],
        "anatomy_list": ["liver"],
        "controllable_anatomy_size": [["hepatic tumor", 0.3], ["liver", 0.5]],
    }
]

TEST_CASE_INFER_DIFFERENT_OUTPUT_TYPE = [
    {
        "bundle_root": "models/maisi_ct_generative",
        "num_output_samples": 1,
        "num_inference_steps": 2,
        "output_size": [256, 256, 256],
        "body_region": ["abdomen"],
        "anatomy_list": ["liver"],
        "image_output_ext": ".dcm",
        "label_output_ext": ".nrrd",
    }
]

TEST_CASE_INFER_ERROR = [
    {
        "bundle_root": "models/maisi_ct_generative",
        "num_output_samples": 1,
        "output_size": [256, 256, 256],
        "body_region": ["head"],
        "anatomy_list": ["colon cancer primaries"],
    },
    "Cannot find body region with given anatomy list.",
]

TEST_CASE_INFER_ERROR_2 = [
    {
        "bundle_root": "models/maisi_ct_generative",
        "num_output_samples": 1,
        "output_size": [256, 256, 256],
        "body_region": ["head_typo"],
        "anatomy_list": ["brain"],
    }
]

TEST_CASE_INFER_ERROR_3 = [
    {
        "bundle_root": "models/maisi_ct_generative",
        "num_output_samples": 1,
        "output_size": [256, 256, 256],
        "body_region": ["head"],
        "anatomy_list": ["brain_typo"],
    }
]

TEST_CASE_INFER_ERROR_4 = [
    {
        "bundle_root": "models/maisi_ct_generative",
        "num_output_samples": 1,
        "output_size": [256, 256, 177],
        "body_region": ["head"],
        "anatomy_list": ["brain"],
    }
]

TEST_CASE_INFER_ERROR_5 = [
    {
        "bundle_root": "models/maisi_ct_generative",
        "num_output_samples": 1,
        "output_size": [256, 256, 256],
        "body_region": ["head"],
        "anatomy_list": ["brain"],
        "controllable_anatomy_size": [["hepatic tumor", 0.3], ["bone lesion", 0.5]],
    }
]

TEST_CASE_INFER_ERROR_6 = [
    {
        "bundle_root": "models/maisi_ct_generative",
        "num_output_samples": 1,
        "output_size": [256, 128, 256],
        "body_region": ["head"],
        "anatomy_list": ["brain"],
        "controllable_anatomy_size": [["hepatic tumor", 0.3], ["bone lesion", 0.5]],
    }
]

TEST_CASE_INFER_ERROR_7 = [
    {
        "bundle_root": "models/maisi_ct_generative",
        "num_output_samples": 1,
        "output_size": [256, 256, 256],
        "body_region": ["chest"],
        "anatomy_list": ["colon", "spleen", "trachea", "left humerus", "sacrum", "heart"],
    },
    "Cannot find body region with given anatomy list.",
]

TEST_CASE_TRAIN = [
    {"bundle_root": "models/maisi_ct_generative", "epochs": 2, "initialize": ["$monai.utils.set_determinism(seed=123)"]}
]


TEST_CASE_TRAIN = [
    {"bundle_root": "models/maisi_ct_generative", "epochs": 2, "initialize": ["$monai.utils.set_determinism(seed=123)"]}
]


def check_workflow(workflow, check_properties: bool = False):
    if check_properties is True:
        check_result = workflow.check_properties()
        if check_result is not None and len(check_result) > 0:
            raise ValueError(f"check properties for workflow failed: {check_result}")
    workflow.run()
    workflow.finalize()


class TestMAISI(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.dataset_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        shutil.rmtree(self.dataset_dir)

    def create_train_dataset(self):
        self.dataset_size = 5
        input_shape = (32, 32, 32, 4)
        mask_shape = (128, 128, 128)
        for s in range(self.dataset_size):
            test_image = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            test_label = np.random.randint(low=0, high=2, size=mask_shape).astype(np.int8)
            image_filename = os.path.join(self.dataset_dir, f"image_{s}.nii.gz")
            label_filename = os.path.join(self.dataset_dir, f"label_{s}.nii.gz")
            nib.save(nib.Nifti1Image(test_image, np.eye(4)), image_filename)
            nib.save(nib.Nifti1Image(test_label, np.eye(4)), label_filename)

    @parameterized.expand([TEST_CASE_TRAIN])
    def test_train_config(self, override):
        self.create_train_dataset()
        train_size = self.dataset_size // 2
        train_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
                "dim": [128, 128, 128],
                "spacing": [1.0, 1.0, 1.0],
                "top_region_index": [0, 1, 0, 0],
                "bottom_region_index": [0, 0, 0, 1],
                "fold": 0,
            }
            for i in range(train_size)
        ]
        override["train_datalist"] = train_datalist

        bundle_root = override["bundle_root"]
        if bundle_root not in sys.path:
            sys.path = [bundle_root] + sys.path
        trainer = create_workflow(
            workflow_type="train",
            config_file=os.path.join(bundle_root, "configs/train.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(trainer, check_properties=False)

    @parameterized.expand(
        [
            TEST_CASE_INFER_1,
            TEST_CASE_INFER_2,
            TEST_CASE_INFER_3,
            TEST_CASE_INFER_WITH_MASK_GENERATION,
            TEST_CASE_INFER_DIFFERENT_OUTPUT_TYPE,
        ]
    )
    def test_infer_config(self, override):
        # update override
        override["output_dir"] = self.output_dir
        bundle_root = override["bundle_root"]
        if bundle_root not in sys.path:
            sys.path = [bundle_root] + sys.path
        workflow = create_workflow(
            workflow_type="infer",
            config_file=os.path.join(bundle_root, "configs/inference.json"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            **override,
        )

        # check_properties=False, need to add monai service properties later
        check_workflow(workflow, check_properties=False)

        # check expected output
        output_files = os.listdir(self.output_dir)
        output_labels = [f for f in output_files if "label" in f]
        output_images = [f for f in output_files if "image" in f]
        self.assertEqual(len(output_labels), override["num_output_samples"])
        self.assertEqual(len(output_images), override["num_output_samples"])

        # check output type and shape
        loader = LoadImage(image_only=True)
        for output_file in output_files:
            output_file_path = os.path.join(self.output_dir, output_file)
            data = loader(output_file_path)
            self.assertEqual(data.shape, tuple(override["output_size"]))
            if "image_output_ext" in override:
                if "image" in output_file:
                    self.assertTrue(output_file.endswith(override["image_output_ext"]))
            elif "label_output_ext" in override:
                if "label" in output_file:
                    self.assertTrue(output_file.endswith(override["label_output_ext"]))
            else:
                self.assertTrue(output_file.endswith(".nii.gz"))

    @parameterized.expand([TEST_CASE_INFER_ERROR, TEST_CASE_INFER_ERROR_7])
    def test_infer_config_error_input(self, override, expected_error):
        # update override
        override["output_dir"] = self.output_dir
        bundle_root = override["bundle_root"]
        if bundle_root not in sys.path:
            sys.path = [bundle_root] + sys.path
        workflow = create_workflow(
            workflow_type="infer",
            config_file=os.path.join(bundle_root, "configs/inference.json"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            **override,
        )
        with self.assertRaises(RuntimeError) as context:
            workflow.run()
        runtime_error = context.exception
        original_exception = runtime_error.__cause__
        self.assertEqual(str(original_exception), expected_error)

    @parameterized.expand(
        [
            TEST_CASE_INFER_ERROR_2,
            TEST_CASE_INFER_ERROR_3,
            TEST_CASE_INFER_ERROR_4,
            TEST_CASE_INFER_ERROR_5,
            TEST_CASE_INFER_ERROR_6,
        ]
    )
    def test_infer_config_valueerror_input(self, override):
        # update override
        override["output_dir"] = self.output_dir
        bundle_root = override["bundle_root"]
        if bundle_root not in sys.path:
            sys.path = [bundle_root] + sys.path
        workflow = create_workflow(
            workflow_type="infer",
            config_file=os.path.join(bundle_root, "configs/inference.json"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            **override,
        )
        with self.assertRaises(RuntimeError) as context:
            workflow.run()
        runtime_error = context.exception
        original_exception = runtime_error.__cause__
        self.assertIsInstance(original_exception, ValueError)


if __name__ == "__main__":
    unittest.main()
