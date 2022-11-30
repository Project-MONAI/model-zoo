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


# This list is used for our CI tests to determine whether a bundle needs to be tested with
# the `verify_data_shape` function in `verify_bundle.py`.
# If a bundle does not need to be tested, please add the bundle name into the list.
exclude_verify_shape_list = ["mednist_gan", "lung_nodule_ct_detection", "pathology_mil_classification"]

# This list is used for our CI tests to determine whether a bundle contains the preferred files.
# If a bundle does not have any of the preferred files, please add the bundle name into the list.
exclude_verify_preferred_files_list = ["pathology_mil_classification"]

# This list is used for our CI tests to determine whether a bundle needs to be tested with
# the `verify_export_torchscript` function in `verify_bundle.py`.
# If a bundle does not support TorchScript, please add the bundle name into the list.
exclude_verify_torchscript_list = [
    "swin_unetr_btcv_segmentation",
    "renalStructures_UNEST_segmentation",
    "wholeBrainSeg_Large_UNEST_segmentation",
    "breast_density_classification",
    "pathology_tumor_detection",
    "mednist_reg",
    "pathology_mil_classification",
]
