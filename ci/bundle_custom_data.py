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


# please add the bundle name into the following list if it does not need to verify data input and output shape.
exclude_verify_shape_list = []

# please add the bundle name into the following list if it does not support torchscript.
exclude_verify_torchscript_list = ["swin_unetr_btcv_segmentation"]

"""
To verify the input and output data shape and data type of network defined in the metadata,
"net_id" and "config_file" are two arguments that need to input.
"net_id" is the ID name of the network component, "config_file" is the filepath (within the bundle)
of the config file to get network definition.
The default values are "network_def" for "net_id" and "configs/inference.json" for "config_file",
if different values are used, please add the bundle (as a key) and a dict in the form of
{"net_id": "", "config_file": ""} (as a value) into the follow dict.

"""
custom_net_config_dict = {"pancreas_ct_dints_segmentation": {"config_file": "configs/inference.yaml"}}
