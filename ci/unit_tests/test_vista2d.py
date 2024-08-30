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

import matplotlib.pyplot as plt
import numpy as np
from monai.bundle import create_workflow
from parameterized import parameterized
from utils import check_workflow

TEST_CASE_TRAIN = [{"bundle_root": "models/vista2d", "mode": "train", "train#trainer#max_epochs": 1}]

TEST_CASE_INFER = [{"bundle_root": "models/vista2d", "mode": "infer"}]


def test_order(test_name1, test_name2):
    def get_order(name):
        if "train" in name:
            return 1
        if "infer" in name:
            return 2
        return 3

    return get_order(test_name1) - get_order(test_name2)


class TestVista2d(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = tempfile.mkdtemp()
        self.tmp_output_dir = os.path.join(self.dataset_dir, "output")
        os.makedirs(self.tmp_output_dir, exist_ok=True)
        self.dataset_size = 5
        input_shape = (256, 256)
        for s in range(self.dataset_size):
            test_image = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            test_label = np.random.randint(low=0, high=2, size=input_shape).astype(np.int8)
            image_filename = os.path.join(self.dataset_dir, f"image_{s}.png")
            label_filename = os.path.join(self.dataset_dir, f"label_{s}.png")
            plt.imsave(image_filename, test_image, cmap="gray")
            plt.imsave(label_filename, test_label, cmap="gray")

        self.bundle_root = "models/vista2d"
        sys.path = [self.bundle_root] + sys.path
        from scripts.workflow import VistaCell

        self.workflow = VistaCell

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    @parameterized.expand([TEST_CASE_INFER])
    def test_infer_config(self, override):
        # update override with dataset dir
        override["dataset#data"] = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{s}.png"),
                "label": os.path.join(self.dataset_dir, f"label_{s}.png"),
            }
            for s in range(self.dataset_size)
        ]
        override["output_dir"] = self.tmp_output_dir
        workflow = create_workflow(
            workflow_name=self.workflow,
            config_file=os.path.join(self.bundle_root, "configs/hyper_parameters.yaml"),
            meta_file=os.path.join(self.bundle_root, "configs/metadata.json"),
            **override,
        )

        # check_properties=False, need to add monai service properties later
        check_workflow(workflow, check_properties=False)

        expected_output_file = os.path.join(self.tmp_output_dir, f"image_{self.dataset_size-1}.tif")
        self.assertTrue(os.path.isfile(expected_output_file))

    @parameterized.expand([TEST_CASE_TRAIN])
    def test_train_config(self, override):
        # update override with dataset dir
        override["train#dataset#data"] = [
            {
                "image": os.path.join(self.dataset_dir, f"image_{s}.png"),
                "label": os.path.join(self.dataset_dir, f"label_{s}.png"),
            }
            for s in range(self.dataset_size)
        ]
        override["dataset#data"] = override["train#dataset#data"]

        workflow = create_workflow(
            workflow_name=self.workflow,
            config_file=os.path.join(self.bundle_root, "configs/hyper_parameters.yaml"),
            meta_file=os.path.join(self.bundle_root, "configs/metadata.json"),
            **override,
        )

        # check_properties=False, need to add monai service properties later
        check_workflow(workflow, check_properties=False)

        # follow up to use trained weights and test eval
        override["mode"] = "eval"
        override["pretrained_ckpt_name"] = "model.pt"
        workflow = create_workflow(
            workflow_name=self.workflow,
            config_file=os.path.join(self.bundle_root, "configs/hyper_parameters.yaml"),
            meta_file=os.path.join(self.bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(workflow, check_properties=False)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = test_order
    unittest.main(testLoader=loader)
