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

import torch
from bundle_custom_data import include_verify_tensorrt_list
from monai.bundle import trt_export
from verify_bundle import _find_bundle_file


def verify_tensorrt(bundle_path: str, net_id: str, config_file: str, precision: str):
    """
    This function is used to verify if the checkpoint is able to torchscript, and
    if "models/model.ts" is provided, it will be checked if is able to be loaded
    successfully.

    """
    trt_model_path = os.path.join(bundle_path, f"models/model_trt_{precision}.ts")
    trt_export(
        net_id=net_id,
        filepath=trt_model_path,
        ckpt_file=os.path.join(bundle_path, "models/model.pt"),
        meta_file=os.path.join(bundle_path, "configs/metadata.json"),
        config_file=os.path.join(bundle_path, config_file),
        precision=precision,
        bundle_root=bundle_path,
        dynamic_batchsize=(1, 1, 1),
    )

    _ = torch.jit.load(trt_model_path)


def verify_bundle(models_path="models"):
    """
    This function is used to verify if the checkpoint is able to export into TensorRT module.
    This function will be updated after the following PR is merged:
    https://github.com/Project-MONAI/MONAI/pull/5986/files

    """
    for bundle in include_verify_tensorrt_list:
        bundle_path = os.path.join(models_path, bundle)
        net_id, inference_file_name = "network_def", _find_bundle_file(
            os.path.join(bundle_path, "configs"), "inference"
        )
        config_file = os.path.join("configs", inference_file_name)
        for precision in ["fp32", "fp16"]:
            verify_tensorrt(bundle_path=bundle_path, net_id=net_id, config_file=config_file, precision=precision)
            print(f"export bundle {bundle} weights into TensorRT module with precision {precision} successfully.")


if __name__ == "__main__":
    verify_bundle()
