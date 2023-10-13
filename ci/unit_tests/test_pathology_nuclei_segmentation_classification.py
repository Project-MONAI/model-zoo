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

import numpy as np
from monai.bundle import ConfigWorkflow
from monai.data import create_test_image_2d
from parameterized import parameterized
from PIL import Image
from utils import check_workflow

TEST_CASE_1 = [  # train, evaluate
    {
        "bundle_root": "models/pathology_nuclei_segmentation_classification",
        "epochs": 1,
        "batch_size": 3,
        "train#dataloader#num_workers": 1,
        "validate#dataloader#num_workers": 1,
    }
]

TEST_CASE_2 = [  # inference
    {"bundle_root": "models/pathology_nuclei_segmentation_classification", "handlers#0#_disabled_": True}
]


class TestNucleiSegCls(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        dataset_size = 10
        # train
        for mode in ["Train", "Test"]:
            train_sample_dir = os.path.join(self.dataset_dir, mode)
            os.makedirs(train_sample_dir)
            for s in range(dataset_size):
                for image_suffix in ["image", "inst_map", "type_map"]:
                    if image_suffix == "image":
                        shape = (256, 256, 3)
                    else:
                        shape = (256, 256, 1)
                    test_image = np.random.randint(low=0, high=2, size=shape).astype(np.int8)
                    image_filename = os.path.join(train_sample_dir, f"{s}_{image_suffix}.npy")
                    np.save(image_filename, test_image)

        # evaluate
        evaluate_sample_dir = os.path.join(self.dataset_dir, "Test")
        for s in range(int(dataset_size / 5)):
            for image_prefix in ["image", "label"]:
                if image_prefix == "image":
                    shape = (256, 256, 3)
                else:
                    shape = (256, 256, 2)
                test_image = np.random.randint(low=0, high=2, size=shape).astype(np.int8)
                image_filename = os.path.join(evaluate_sample_dir, f"{image_prefix}_{s}.npy")
                np.save(image_filename, test_image)

        # inference
        self.infer_sample_dir = os.path.join(self.dataset_dir, "Images")
        os.makedirs(self.infer_sample_dir)
        for s in range(int(dataset_size / 5)):
            img, _ = create_test_image_2d(1000, 1000)
            im = Image.fromarray(img).convert("RGB")
            # img = Image.new("RGB", (1000, 1000))
            image_filename = os.path.join(self.infer_sample_dir, f"image_{s}.png")
            im.save(image_filename, "PNG")

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
        override["dataset_dir"] = self.infer_sample_dir
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
