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

import shutil
import tempfile

from enums import ContextKeys
from test_config_utils import TestBundleConfigs, produce_dataset

TEST_CASE = {
    # all config filenames that need to be tested
    ContextKeys.TRAIN: "train.json",
    ContextKeys.MGPU_TRAIN: "multi_gpu_train.json",
    ContextKeys.EVAL: "evaluate.json",
    ContextKeys.MGPU_EVAL: "multi_gpu_evaluate.json",
    ContextKeys.INFER: "inference.json",
    # all dataset related info
    ContextKeys.DATASET_SIZE: 10,
    ContextKeys.INPUT_SHAPE: (64, 64, 64),
    # override info
    ContextKeys.TRAIN_OVERRIDE: {
        "images": "$list(sorted(glob.glob(@dataset_dir + '/image_*.nii.gz')))",
        "labels": "$list(sorted(glob.glob(@dataset_dir + '/label_*.nii.gz')))",
        "epochs": 1,
        "train#dataset#cache_rate": 0.0,
        "validate#dataset#cache_rate": 0.0,
        "train#dataloader#num_workers": 1,
        "validate#dataloader#num_workers": 1,
        "train#random_transforms#0#spatial_size": [32, 32, 32],
        "validate#handlers#-1#key_metric_filename": "test_model_1.pt",
    },
    ContextKeys.INFER_OVERRIDE: {"datalist": "$list(sorted(glob.glob(@dataset_dir + '/image_*.nii.gz')))"},
    # determinism related info
    ContextKeys.TRAIN_DETERMINISM_OVERRIDE: {
        "images": "$list(sorted(glob.glob(@dataset_dir + '/image_*.nii.gz')))",
        "labels": "$list(sorted(glob.glob(@dataset_dir + '/label_*.nii.gz')))",
        "epochs": 1,
        "train#dataset#cache_rate": 0.0,
        "validate#dataset#cache_rate": 0.0,
        "train#dataloader#num_workers": 1,
        "validate#dataloader#num_workers": 1,
        "train#random_transforms#0#spatial_size": [32, 32, 32],
        "validate#handlers#-1#key_metric_filename": "test_model_2.pt",
    },
    ContextKeys.DETERMINISM_WEIGHTS_1: "test_model_1.pt",
    ContextKeys.DETERMINISM_WEIGHTS_2: "test_model_2.pt",
}


def test_bundle_configs(bundle_root):
    TEST_CASE[ContextKeys.DATASET_DIR] = tempfile.mkdtemp()
    produce_dataset(test_context=TEST_CASE)
    config_tests = TestBundleConfigs(bundle_root=bundle_root, test_context=TEST_CASE)
    config_tests.test_train_config()
    config_tests.test_inference_config()
    config_tests.test_train_determinism()

    shutil.rmtree(TEST_CASE[ContextKeys.DATASET_DIR])
