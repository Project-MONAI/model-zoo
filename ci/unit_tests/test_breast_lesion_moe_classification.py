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

import numpy as np
from monai.bundle import ConfigWorkflow
from parameterized import parameterized
from utils import check_workflow

# The bundle ships synthetic embeddings (seeded standard-normal vectors) in
# sample_data/. The "train" case runs one epoch from random init and the
# offline "inference" case runs with random weights (CheckpointLoader
# disabled), so both only prove the pipeline runs end to end -- predictions
# on synthetic embeddings are meaningless. The checkpoint-enabled inference
# case additionally loads the delivered models/model.pt (large_files.yml)
# to check checkpoint/network compatibility; it is skipped when the
# checkpoint has not been downloaded.
TEST_CASE_1 = [  # train, evaluate
    {
        "bundle_root": "models/breast_lesion_moe_classification",
        "dataset_dir": "PLACEHOLDER",
        "train#dataloader#num_workers": 0,
        "validate#dataloader#num_workers": 0,
        "num_epochs": 1,
        "val_interval": 1,
    }
]

TEST_CASE_2 = [  # inference
    {
        "bundle_root": "models/breast_lesion_moe_classification",
        "dataloader#num_workers": 0,
        "handlers#0#_disabled_": True,  # do not load weights
    }
]

# Delivered checkpoint (large_files.yml): models/model.pt under the bundle
# root. Absent from a plain repo checkout, so the checkpoint-enabled case
# below skips unless it has been downloaded.
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "breast_lesion_moe_classification",
    "models",
    "model.pt",
)

TEST_CASE_3 = [  # inference with the delivered checkpoint
    {"bundle_root": "models/breast_lesion_moe_classification", "dataloader#num_workers": 0}
]


def test_order(test_name1, test_name2):
    def get_order(name):
        if "train" in name:
            return 1
        if "eval" in name:
            return 2
        if "infer" in name:
            return 3
        return 4

    return get_order(test_name1) - get_order(test_name2)


class TestBreastLesionMoEClassification(unittest.TestCase):
    def setUp(self):
        # Make the bundle's `scripts` package importable, as in other bundle
        # tests -- the repo's test runner does not set PYTHONPATH.
        bundle_root = "models/breast_lesion_moe_classification"
        if bundle_root not in sys.path:
            sys.path.insert(0, bundle_root)
        self.dataset_dir = tempfile.mkdtemp()
        dataset_size = 8
        rng = np.random.default_rng(0)
        train_entries = []
        val_entries = []
        for split, entries in (("train", train_entries), ("val", val_entries)):
            for s in range(dataset_size):
                entry = {"image_id": f"{split}_{s:03d}", "label": int(s % 3)}
                for expert, dim in (("uni", 1024), ("virchow2", 2560), ("phikon_v2_ft", 1024)):
                    arr = rng.normal(0.0, 0.03, size=(dim,)).astype(np.float32)
                    fname = os.path.join(self.dataset_dir, f"{split}_{s:03d}__{expert}.npy")
                    np.save(fname, arr)
                    entry[expert] = fname
                entries.append(entry)
        import json

        with open(os.path.join(self.dataset_dir, "train_datalist.json"), "w") as f:
            json.dump(train_entries, f)
        with open(os.path.join(self.dataset_dir, "val_datalist.json"), "w") as f:
            json.dump(val_entries, f)

    def tearDown(self):
        shutil.rmtree(self.dataset_dir)

    @parameterized.expand([TEST_CASE_1])
    def test_train_eval_config(self, override):
        override["dataset_dir"] = self.dataset_dir
        # keep checkpoints / logs out of the bundle directory
        override["ckpt_dir"] = os.path.join(self.dataset_dir, "ckpt")
        override["output_dir"] = os.path.join(self.dataset_dir, "log")
        bundle_root = override["bundle_root"]
        train_file = os.path.join(bundle_root, "configs/train.json")

        trainer = ConfigWorkflow(
            workflow_type="train",
            config_file=train_file,
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(trainer, check_properties=True)

    @parameterized.expand([TEST_CASE_2])
    def test_infer_config(self, override):
        bundle_root = override["bundle_root"]

        inferrer = ConfigWorkflow(
            workflow_type="infer",
            config_file=os.path.join(bundle_root, "configs/inference.json"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
            **override,
        )
        check_workflow(inferrer, check_properties=True)

    @parameterized.expand([TEST_CASE_3])
    @unittest.skipUnless(os.path.exists(CHECKPOINT_PATH), "delivered checkpoint (models/model.pt) not downloaded")
    def test_infer_config_with_checkpoint(self, override):
        # keep predictions out of the bundle directory
        override["output_dir"] = os.path.join(self.dataset_dir, "infer_ckpt")
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
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = test_order
    unittest.main(testLoader=loader)
