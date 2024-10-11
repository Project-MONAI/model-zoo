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

TEST_CASE_INFER = [
    {
        "bundle_root": "models/vista3d",
        "input_dict": {"label_prompt": [25], "points": [[123, 212, 151]], "point_labels": [1]},
        "patch_size": [32, 32, 32],
        "checkpointloader#_disabled_": True,  # do not load weights"
        "initialize": ["$monai.utils.set_determinism(seed=123)"],
    }
]
TEST_CASE_INFER_STR_PROMPT = [
    {
        "bundle_root": "models/vista3d",
        "input_dict": {"label_prompt": ["spleen"], "points": [[123, 212, 151]], "point_labels": [1]},
        "patch_size": [32, 32, 32],
        "checkpointloader#_disabled_": True,  # do not load weights"
        "initialize": ["$monai.utils.set_determinism(seed=123)"],
    }
]
TEST_CASE_INFER_MULTI_PROMPT = [
    {
        "bundle_root": "models/vista3d",
        "input_dict": {"label_prompt": [25, 24, 1]},
        "patch_size": [32, 32, 32],
        "checkpointloader#_disabled_": True,  # do not load weights"
        "initialize": ["$monai.utils.set_determinism(seed=123)"],
    }
]
TEST_CASE_INFER_MULTI_STR_PROMPT = [
    {
        "bundle_root": "models/vista3d",
        "input_dict": {"label_prompt": ["hepatic vessel", "pancreatic tumor", "liver"]},
        "patch_size": [32, 32, 32],
        "checkpointloader#_disabled_": True,  # do not load weights"
        "initialize": ["$monai.utils.set_determinism(seed=123)"],
    }
]
TEST_CASE_INFER_MULTI_NEW_STR_PROMPT = [
    {
        "bundle_root": "models/vista3d",
        "input_dict": {"label_prompt": ["new class 1"], "points": [[123, 212, 151]], "point_labels": [1]},
        "patch_size": [32, 32, 32],
        "checkpointloader#_disabled_": True,  # do not load weights"
        "initialize": ["$monai.utils.set_determinism(seed=123)"],
    }
]
TEST_CASE_INFER_SUBCLASS = [
    {
        "bundle_root": "models/vista3d",
        "input_dict": {"label_prompt": [2, 20, 21]},
        "patch_size": [32, 32, 32],
        "checkpointloader#_disabled_": True,  # do not load weights"
        "initialize": ["$monai.utils.set_determinism(seed=123)"],
    }
]
TEST_CASE_INFER_NO_PROMPT = [
    {
        "bundle_root": "models/vista3d",
        "input_dict": {},  # put an empty dict, and will add an image in the test function
        "patch_size": [32, 32, 32],
        "checkpointloader#_disabled_": True,  # do not load weights"
        "initialize": ["$monai.utils.set_determinism(seed=123)"],
    }
]
TEST_CASE_EVAL = [
    {
        "bundle_root": "models/vista3d",
        "patch_size": [32, 32, 32],
        "initialize": ["$monai.utils.set_determinism(seed=123)"],
    }
]
TEST_CASE_TRAIN = [
    {
        "bundle_root": "models/vista3d",
        "patch_size": [32, 32, 32],
        "epochs": 2,
        "val_interval": 1,
        "initialize": ["$monai.utils.set_determinism(seed=123)"],
    }
]
TEST_CASE_TRAIN_CONTINUAL = [
    {
        "bundle_root": "models/vista3d",
        "patch_size": [32, 32, 32],
        "epochs": 2,
        "val_interval": 1,
        "initialize": ["$monai.utils.set_determinism(seed=123)"],
        "finetune": False,
    }
]
TEST_CASE_ERROR_PROMPTS = [
    [
        {
            "bundle_root": "models/vista3d",
            "input_dict": {},
            "patch_size": [32, 32, 32],
            "checkpointloader#_disabled_": True,  # do not load weights"
            "everything_labels": None,
            "initialize": ["$monai.utils.set_determinism(seed=123)"],
            "error": "Prompt must be given for inference.",
        }
    ],
    [
        {
            "bundle_root": "models/vista3d",
            "input_dict": {"label_prompt": [[25, 26, 27]]},
            "patch_size": [32, 32, 32],
            "checkpointloader#_disabled_": True,  # do not load weights"
            "initialize": ["$monai.utils.set_determinism(seed=123)"],
            "error": "Label prompt must be a list of single scalar, [1,2,3,4,...,].",
        }
    ],
    [
        {
            "bundle_root": "models/vista3d",
            "input_dict": {"label_prompt": 25},
            "patch_size": [32, 32, 32],
            "checkpointloader#_disabled_": True,  # do not load weights"
            "initialize": ["$monai.utils.set_determinism(seed=123)"],
            "error": "Label prompt must be a list, [1,2,3,4,...,].",
        }
    ],
    [
        {
            "bundle_root": "models/vista3d",
            "input_dict": {"label_prompt": [256]},
            "patch_size": [32, 32, 32],
            "checkpointloader#_disabled_": True,  # do not load weights"
            "initialize": ["$monai.utils.set_determinism(seed=123)"],
            "error": "Current bundle only supports label prompt smaller than 255.",
        }
    ],
    [
        {
            "bundle_root": "models/vista3d",
            "input_dict": {"label_prompt": [25], "points": [[123, 212, 151]]},
            "patch_size": [32, 32, 32],
            "checkpointloader#_disabled_": True,  # do not load weights"
            "initialize": ["$monai.utils.set_determinism(seed=123)"],
            "error": "Point labels must be given if points are given.",
        }
    ],
    [
        {
            "bundle_root": "models/vista3d",
            "input_dict": {"label_prompt": [25], "point_labels": [1]},
            "patch_size": [32, 32, 32],
            "checkpointloader#_disabled_": True,  # do not load weights"
            "initialize": ["$monai.utils.set_determinism(seed=123)"],
            "error": "Points must be given if point labels are given.",
        }
    ],
    [
        {
            "bundle_root": "models/vista3d",
            "input_dict": {"label_prompt": [25], "points": [[1, 123, 212, 151]], "point_labels": [1]},
            "patch_size": [32, 32, 32],
            "checkpointloader#_disabled_": True,  # do not load weights"
            "initialize": ["$monai.utils.set_determinism(seed=123)"],
            "error": "Points must be three dimensional (x,y,z) in the shape of [[x,y,z],...,[x,y,z]].",
        }
    ],
    [
        {
            "bundle_root": "models/vista3d",
            "input_dict": {"label_prompt": [25], "points": [[[123, 212, 151]]], "point_labels": [1]},
            "patch_size": [32, 32, 32],
            "checkpointloader#_disabled_": True,  # do not load weights"
            "initialize": ["$monai.utils.set_determinism(seed=123)"],
            "error": "Points must be three dimensional (x,y,z) in the shape of [[x,y,z],...,[x,y,z]].",
        }
    ],
    [
        {
            "bundle_root": "models/vista3d",
            "input_dict": {"label_prompt": [25], "points": [[123, 212, 151]], "point_labels": [1, 1]},
            "patch_size": [32, 32, 32],
            "checkpointloader#_disabled_": True,  # do not load weights"
            "initialize": ["$monai.utils.set_determinism(seed=123)"],
            "error": "Points must match point labels.",
        }
    ],
    [
        {
            "bundle_root": "models/vista3d",
            "input_dict": {"label_prompt": [1], "points": [[123, 212, 151]], "point_labels": [-2]},
            "patch_size": [32, 32, 32],
            "checkpointloader#_disabled_": True,  # do not load weights"
            "initialize": ["$monai.utils.set_determinism(seed=123)"],
            "error": "Point labels can only be -1,0,1 and 2,3 for special flags.",
        }
    ],
    [
        {
            "bundle_root": "models/vista3d",
            "input_dict": {"label_prompt": [25, 26], "points": [[123, 212, 151]], "point_labels": [1]},
            "patch_size": [32, 32, 32],
            "checkpointloader#_disabled_": True,  # do not load weights"
            "initialize": ["$monai.utils.set_determinism(seed=123)"],
            "error": "Label prompt can only be a single object if provided with point prompts.",
        }
    ],
    [
        {
            "bundle_root": "models/vista3d",
            "input_dict": {"label_prompt": [16, 25, 26]},
            "patch_size": [32, 32, 32],
            "checkpointloader#_disabled_": True,  # do not load weights"
            "initialize": ["$monai.utils.set_determinism(seed=123)"],
            "error": "Undefined label prompt detected. Provide point prompts for zero-shot.",
        }
    ],
    [
        {
            "bundle_root": "models/vista3d",
            "input_dict": {"label_prompt": [136]},
            "patch_size": [32, 32, 32],
            "checkpointloader#_disabled_": True,  # do not load weights"
            "initialize": ["$monai.utils.set_determinism(seed=123)"],
            "error": "Undefined label prompt detected. Provide point prompts for zero-shot.",
        }
    ],
]


def test_order(test_name1, test_name2):
    def get_order(name):
        if "train_config" in name:
            return 1
        if "train_continual" in name:
            return 2
        if "eval" in name:
            return 3
        return 4

    return get_order(test_name1) - get_order(test_name2)


class TestVista3d(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        self.dataset_size = 5
        input_shape = (64, 64, 64)
        for s in range(self.dataset_size):
            test_image = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            test_label = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            image_filename = os.path.join(self.dataset_dir, f"image_{s}.nii.gz")
            label_filename = os.path.join(self.dataset_dir, f"label_{s}.nii.gz")
            nib.save(nib.Nifti1Image(test_image, np.eye(4)), image_filename)
            nib.save(nib.Nifti1Image(test_label, np.eye(4)), label_filename)

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    @parameterized.expand([TEST_CASE_TRAIN])
    def test_train_config(self, override):
        train_size = self.dataset_size // 2
        train_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
            }
            for i in range(train_size)
        ]
        val_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
            }
            for i in range(train_size, self.dataset_size)
        ]
        override["train_datalist"] = train_datalist
        override["val_datalist"] = val_datalist

        bundle_root = override["bundle_root"]
        if bundle_root not in sys.path:
            sys.path = [bundle_root] + sys.path
        trainer = ConfigWorkflow(
            workflow_type="train",
            config_file=os.path.join(bundle_root, "configs/train.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(trainer, check_properties=False)

    @parameterized.expand([TEST_CASE_EVAL])
    def test_eval_config(self, override):
        train_size = self.dataset_size // 2
        train_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
            }
            for i in range(train_size)
        ]
        val_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
            }
            for i in range(train_size, self.dataset_size)
        ]
        override["train_datalist"] = train_datalist
        override["val_datalist"] = val_datalist
        bundle_root = override["bundle_root"]
        if bundle_root not in sys.path:
            sys.path = [bundle_root] + sys.path
        config_files = [
            os.path.join(bundle_root, "configs/train.json"),
            os.path.join(bundle_root, "configs/train_continual.json"),
            os.path.join(bundle_root, "configs/evaluate.json"),
        ]
        trainer = ConfigWorkflow(
            workflow_type="train",
            config_file=config_files,
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(trainer, check_properties=False)

    @parameterized.expand([TEST_CASE_TRAIN_CONTINUAL])
    def test_train_continual_config(self, override):
        train_size = self.dataset_size // 2
        train_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
            }
            for i in range(train_size)
        ]
        val_datalist = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{i}.nii.gz"),
                "label": os.path.join(self.dataset_dir, f"label_{i}.nii.gz"),
            }
            for i in range(train_size, self.dataset_size)
        ]
        override["train_datalist"] = train_datalist
        override["val_datalist"] = val_datalist

        bundle_root = override["bundle_root"]
        if bundle_root not in sys.path:
            sys.path = [bundle_root] + sys.path
        trainer = ConfigWorkflow(
            workflow_type="train",
            config_file=[
                os.path.join(bundle_root, "configs/train.json"),
                os.path.join(bundle_root, "configs/train_continual.json"),
            ],
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(trainer, check_properties=False)

    @parameterized.expand(
        [
            TEST_CASE_INFER,
            TEST_CASE_INFER_MULTI_PROMPT,
            TEST_CASE_INFER_NO_PROMPT,
            TEST_CASE_INFER_SUBCLASS,
            TEST_CASE_INFER_STR_PROMPT,
            TEST_CASE_INFER_MULTI_STR_PROMPT,
            TEST_CASE_INFER_MULTI_NEW_STR_PROMPT,
        ]
    )
    def test_infer_config(self, override):
        # update input_dict with dataset dir
        input_dict = override["input_dict"]
        input_dict["image"] = os.path.join(self.dataset_dir, "image_0.nii.gz")
        override["input_dict"] = input_dict

        bundle_root = override["bundle_root"]
        if bundle_root not in sys.path:
            sys.path = [bundle_root] + sys.path

        inferrer = ConfigWorkflow(
            workflow_type="infer",
            config_file=os.path.join(bundle_root, "configs/inference.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        # check_properties=False because this bundle does not have some required properties such as dataset_dir
        check_workflow(inferrer, check_properties=False)

    @parameterized.expand(
        [TEST_CASE_INFER, TEST_CASE_INFER_MULTI_PROMPT, TEST_CASE_INFER_NO_PROMPT, TEST_CASE_INFER_SUBCLASS]
    )
    def test_batch_infer_config(self, override):
        # update input_dict with dataset dir
        params = override.copy()
        params.pop("input_dict", None)
        params["input_dir"] = self.dataset_dir
        params["input_suffix"] = "image_*.nii.gz"

        bundle_root = override["bundle_root"]
        if bundle_root not in sys.path:
            sys.path = [bundle_root] + sys.path
        config_files = [
            os.path.join(bundle_root, "configs/inference.json"),
            os.path.join(bundle_root, "configs/batch_inference.json"),
        ]
        inferrer = ConfigWorkflow(
            workflow_type="infer",
            config_file=config_files,
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **params,
        )
        # check_properties=False because this bundle does not have some required properties such as dataset_dir
        check_workflow(inferrer, check_properties=False)

    @parameterized.expand(TEST_CASE_ERROR_PROMPTS)
    def test_error_prompt_infer_config(self, override):
        # update input_dict with dataset dir
        input_dict = override["input_dict"]
        input_dict["image"] = os.path.join(self.dataset_dir, "image_0.nii.gz")
        override["input_dict"] = input_dict

        bundle_root = override["bundle_root"]
        if bundle_root not in sys.path:
            sys.path = [bundle_root] + sys.path

        inferrer = ConfigWorkflow(
            workflow_type="infer",
            config_file=os.path.join(bundle_root, "configs/inference.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        inferrer.initialize()
        with self.assertRaises(RuntimeError) as context:
            inferrer.run()
        runtime_error = context.exception
        original_exception = runtime_error.__cause__
        self.assertEqual(str(original_exception), override["error"])

    @parameterized.expand([TEST_CASE_INFER])
    def test_labels_dict(self, override):
        bundle_root = override["bundle_root"]
        label_dict_file = os.path.join(bundle_root, "docs/labels.json")
        if not os.path.isfile(label_dict_file):
            raise ValueError(f"labels.json not found in {bundle_root}")
        with open(label_dict_file) as f:
            _ = json.load(f)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = test_order
    unittest.main(testLoader=loader)
