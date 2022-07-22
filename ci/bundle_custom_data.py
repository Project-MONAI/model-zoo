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
exclude_verify_shape_list = ["mednist_gan"]

# This list is used for our CI tests to determine whether a bundle needs to be tested with
# the `verify_export_torchscript` function in `verify_bundle.py`.
# If a bundle does not support TorchScript, please add the bundle name into the list.
exclude_verify_torchscript_list = ["swin_unetr_btcv_segmentation"]

# This dict is used for our CI tests to override the values of the arguments "net_id" and "config_file"
# for the `verify_data_shape` and `verify_export_torchscript` functions.
# The default values are "network_def" for "net_id" and `configs/inference.json` for `config_file`.
# please add the bundle name (as a key) and a dict in the form of {"net_id": "", "config_file": ""} (as a value)
# into the following dict.
custom_net_config_dict = {"pancreas_ct_dints_segmentation": {"config_file": "configs/inference.yaml"}}
