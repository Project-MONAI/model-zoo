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
from bundle_custom_data import include_verify_onnx_tensorrt_dict, include_verify_tensorrt_dict
from download_latest_bundle import download_latest_bundle
from monai.bundle import trt_export
from verify_bundle import _find_bundle_file


def verify_tensorrt(export_context):
    """
    This function is used to verify if the checkpoint is able to export into TensorRT model, and
    the exported model will be checked if it is able to be loaded successfully.

    """
    bundle_root = export_context["bundle_root"]
    precision = export_context["precision"]
    trt_model_path = os.path.join(bundle_root, f"models/model_trt_{precision}.ts")
    export_context["filepath"] = trt_model_path
    try:
        trt_export(**export_context)
    except Exception as e:
        print(f"'trt_export' failed with error: {e}")
        raise
    try:
        torch.jit.load(trt_model_path)
    except Exception as e:
        print(f"load TensorRT model {trt_model_path} failed with error: {e}")
        raise


def get_export_required_files(bundle: str, download_path: str, use_onnx: bool = False, use_trace: bool = False):
    download_latest_bundle(bundle_name=bundle, models_path="models", download_path=download_path)
    bundle_path = os.path.join(download_path, bundle)
    net_id, inference_file_name = "network_def", _find_bundle_file(os.path.join(bundle_path, "configs"), "inference")
    config_file = os.path.join("configs", inference_file_name)
    ckpt_file = os.path.join(bundle_path, "models/model.pt")
    meta_file = os.path.join(bundle_path, "configs/metadata.json")
    config_file = os.path.join(bundle_path, config_file)

    export_context = {
        "bundle_root": bundle_path,
        "net_id": net_id,
        "config_file": config_file,
        "ckpt_file": ckpt_file,
        "meta_file": meta_file,
        "use_onnx": use_onnx,
        "use_trace": use_trace,
    }
    return export_context


def verify_all_tensorrt_bundles(download_path="download"):
    """
    This function is used to verify all bundles that support TensorRT.

    """

    for bundle in include_verify_tensorrt_dict:
        print(f"start verifying bundle {bundle} into TensorRT module.")
        export_context = get_export_required_files(bundle, download_path)
        extra_parameters = include_verify_tensorrt_dict[bundle]
        export_context.update(extra_parameters)
        for precision in ["fp32", "fp16"]:
            export_context["precision"] = precision
            try:
                verify_tensorrt(export_context)
                print(f"export bundle {bundle} with precision {precision} successfully.")
            except BaseException:
                print(f"export bundle {bundle} with precision {precision} failed.")
                raise
    print("all TensorRT supported bundles are verified correctly.")


def verify_all_onnx_tensorrt_bundles(download_path="download"):
    """
    This function is used to verify all bundles that support ONNX-TensorRT.

    """

    for bundle in include_verify_onnx_tensorrt_dict:
        print(f"start verifying export bundle {bundle} into ONNX-TensorRT module.")
        extra_parameters = include_verify_onnx_tensorrt_dict[bundle]
        export_context = get_export_required_files(bundle, download_path, use_onnx=True, use_trace=True)
        export_context.update(extra_parameters)
        for precision in ["fp32", "fp16"]:
            export_context["precision"] = precision
            try:
                verify_tensorrt(export_context)
                print(f"export bundle {bundle} with precision {precision} successfully.")
            except BaseException:
                print(f"export bundle {bundle} with precision {precision} failed.")
                raise
    print("all ONNX-TensorRT supported bundles are verified correctly.")


if __name__ == "__main__":
    verify_all_tensorrt_bundles()
    verify_all_onnx_tensorrt_bundles()
