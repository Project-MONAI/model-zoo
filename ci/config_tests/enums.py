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


class ContextKeys:
    # filename keys
    TRAIN = "train"
    MGPU_TRAIN = "mgpu_train"
    EVAL = "evaluate"
    MGPU_EVAL = "mgpu_evaluate"
    INFER = "inference"
    # override keys
    TRAIN_OVERRIDE = "train_override"
    TRAIN_DETERMINISM_OVERRIDE = "train_determinism_override"
    INFER_OVERRIDE = "inference_override"
    # dataset keys
    DATASET_DIR = "dataset_dir"
    DATASET_SIZE = "dataset_size"
    INPUT_SHAPE = "input_shape"
    # image type keys
    DATA_TYPE = "data_type"
    NIBABEL = "nibabel"
    # determinism keys
    DETERMINISM_WEIGHTS_1 = "determinism_weights_1"
    DETERMINISM_WEIGHTS_2 = "determinism_weights_2"
