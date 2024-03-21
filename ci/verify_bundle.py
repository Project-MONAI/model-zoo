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

import argparse
import os
import sys
from typing import List

import torch
from bundle_custom_data import (
    exclude_verify_preferred_files_list,
    exclude_verify_shape_list,
    exclude_verify_torchscript_list,
)
from monai.bundle import ckpt_export, verify_metadata, verify_net_in_out
from monai.bundle.config_parser import ConfigParser
from utils import download_large_files, get_json_dict

# files that must be included in a bundle
necessary_files_list = ["configs/metadata.json", "LICENSE"]
# files that are preferred to be included in a bundle
preferred_files_list = ["models/model.pt", "configs/inference.json"]
# keys that must be included in metadata #TODO: should add it into MONAI Label and NVFlare required properties
metadata_keys_list = ["name"]


def _find_bundle_file(root_dir: str, file: str, suffix=("json", "yaml", "yml")):
    # find bundle file with possible suffix
    file_name = None
    for name in suffix:
        full_name = f"{file}.{name}"
        if full_name in os.listdir(root_dir):
            file_name = full_name

    return file_name


def _get_weights_names(bundle: str):
    # TODO: this function is temporarily used. It should be replaced by detailed config tests.
    if bundle == "brats_mri_generative_diffusion":
        return "model_autoencoder.pt", "model_autoencoder.ts"
    if bundle == "brats_mri_axial_slices_generative_diffusion":
        return "model_autoencoder.pt", None
    return "model.pt", "model.ts"


def _get_net_id(bundle: str):
    # TODO: this function is temporarily used. It should be replaced by detailed config tests.
    if bundle == "brats_mri_generative_diffusion":
        return "autoencoder_def"
    if bundle == "brats_mri_axial_slices_generative_diffusion":
        return "autoencoder_def"
    return "network_def"


def _check_missing_keys(file_name: str, bundle_path: str, keys_list: List):
    config = ConfigParser.load_config_file(os.path.join(bundle_path, "configs", file_name))
    missing_keys = []
    for key in keys_list:
        if key not in config:
            missing_keys.append(key)

    if len(missing_keys) > 0:
        raise ValueError(f"missing key(s): {str(missing_keys)} in {file_name}.")

    return config


def verify_bundle_directory(models_path: str, bundle_name: str):
    """
    According to [MONAI Bundle Specification](https://docs.monai.io/en/latest/mb_specification.html),
    as well as the requirements of model zoo, some files are necessary with the bundle. For example:
    "models/model.pt", "configs/metadata.json".
    This function is used to verify these files. For bundles that contain the download links for large
    files, the links should be saved in "large_files.yml" (or .json, .yaml).
    All large files (if exist) will be downloaded before verification.

    """

    bundle_path = os.path.join(models_path, bundle_name)

    # download large files (if exist) first
    large_file_name = _find_bundle_file(bundle_path, "large_files")
    if large_file_name is not None:
        try:
            download_large_files(bundle_path=bundle_path, large_file_name=large_file_name)
        except Exception as e:
            raise ValueError(f"Download large files in {bundle_path} error.") from e

    # verify necessary files are included
    for file in necessary_files_list:
        if not os.path.exists(os.path.join(bundle_path, file)):
            raise ValueError(f"necessary file {file} is not existing.")

    # verify preferred files are included
    if bundle_name not in exclude_verify_preferred_files_list:
        for file in preferred_files_list:
            if file == "configs/inference.json":
                # inference config file may have different suffix
                inference_file_name = _find_bundle_file(os.path.join(bundle_path, "configs"), "inference")
                if inference_file_name is None:
                    raise ValueError("inference config file is not existing.")
            else:
                if not os.path.exists(os.path.join(bundle_path, file)):
                    raise ValueError(f"necessary file {file} is not existing.")


def verify_bundle_keys(models_path: str, bundle_name: str):
    """
    This function is partially replaced by `check_properties` in unit tests.

    """
    bundle_path = os.path.join(models_path, bundle_name)

    # verify metadata
    _ = _check_missing_keys(file_name="metadata.json", bundle_path=bundle_path, keys_list=metadata_keys_list)


def verify_version_changes(models_path: str, bundle_name: str):
    """
    This function is used to verify if "version" and "changelog" are correct in "configs/metadata.json".
    In addition, if changing an existing bundle, a new version number should be provided.

    """

    bundle_path = os.path.join(models_path, bundle_name)

    meta_file_path = os.path.join(bundle_path, "configs/metadata.json")
    metadata = get_json_dict(meta_file_path)
    if "version" not in metadata:
        raise ValueError(f"'version' is missing in configs/metadata.json of bundle: {bundle_name}.")
    if "changelog" not in metadata:
        raise ValueError(f"'changelog' is missing in configs/metadata.json of bundle: {bundle_name}.")

    # version number should be in changelog
    latest_version = metadata["version"]
    if latest_version not in metadata["changelog"].keys():
        raise ValueError(
            f"version number: {latest_version} is missing in 'changelog' in configs/metadata.json of bundle: {bundle_name}."
        )

    # If changing an existing bundle, a new version number should be provided
    model_info_path = os.path.join(models_path, "model_info.json")
    model_info = get_json_dict(model_info_path)
    bundle_name_with_version = f"{bundle_name}_v{latest_version}"
    if bundle_name_with_version in model_info.keys():
        raise ValueError(
            f"version number: {latest_version} is already used of bundle: {bundle_name}. Please change it."
        )


def verify_metadata_format(bundle_path: str):
    """
    This function is used to verify the metadata format.

    """
    verify_metadata(
        meta_file=os.path.join(bundle_path, "configs/metadata.json"),
        filepath=os.path.join(bundle_path, "eval/schema.json"),
    )


def verify_data_shape(bundle_path: str, net_id: str, config_file: str):
    """
    This function is used to verify the data shape of network.

    """
    verify_net_in_out(
        net_id=net_id,
        meta_file=os.path.join(bundle_path, "configs/metadata.json"),
        config_file=os.path.join(bundle_path, config_file),
        bundle_root=bundle_path,
    )


def verify_torchscript(
    bundle_path: str, net_id: str, config_file: str, model_name: str = "model.pt", ts_name: str = "model.ts"
):
    """
    This function is used to verify if the checkpoint is able to export into torchscript model, and
    if "models/model.ts" is provided, it will be checked if it is able to be loaded
    successfully.

    """
    ckpt_export(
        net_id=net_id,
        filepath=os.path.join(bundle_path, "models/verify_model.ts"),
        ckpt_file=os.path.join(bundle_path, "models", model_name),
        meta_file=os.path.join(bundle_path, "configs/metadata.json"),
        config_file=os.path.join(bundle_path, config_file),
        bundle_root=bundle_path,
    )
    print("export weights into TorchScript module successfully.")

    ts_model_path = os.path.join(bundle_path, "models", ts_name)
    if os.path.exists(ts_model_path):
        torch.jit.load(ts_model_path)
        print("Provided TorchScript module is verified correctly.")


def verify(bundle, models_path="models", mode="full"):
    print(f"start verifying {bundle}:")
    # add bundle path to ensure custom code can be used
    sys.path = [os.path.join(models_path, bundle)] + sys.path
    # verify bundle directory
    verify_bundle_directory(models_path, bundle)
    print("directory is verified correctly.")
    # verify bundle keys
    verify_bundle_keys(models_path, bundle)
    print("keys are verified correctly.")
    if mode != "regular":
        # verify version, changelog
        verify_version_changes(models_path, bundle)
        print("version and changelog are verified correctly.")
    # verify metadata format and data
    bundle_path = os.path.join(models_path, bundle)
    verify_metadata_format(bundle_path)
    print("metadata format is verified correctly.")

    if mode in ["min", "regular"]:
        return

    # The following are optional tests and require GPU
    net_id = _get_net_id(bundle)
    inference_file_name = _find_bundle_file(os.path.join(bundle_path, "configs"), "inference")
    config_file = os.path.join("configs", inference_file_name)

    if bundle in exclude_verify_shape_list:
        print(f"skip verifying the data shape of bundle: {bundle}.")
    else:
        verify_data_shape(bundle_path, net_id, config_file)
        print("data shape is verified correctly.")

    if bundle in exclude_verify_torchscript_list:
        print(f"bundle: {bundle} does not support torchscript, skip verifying.")
    else:
        model_name, ts_name = _get_weights_names(bundle=bundle)
        verify_torchscript(bundle_path, net_id, config_file, model_name, ts_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-b", "--b", type=str, help="bundle name.")
    parser.add_argument("-p", "--p", type=str, default="models", help="models path.")
    parser.add_argument("-m", "--mode", type=str, default="full", help="verify bundle mode (full/min).")
    args = parser.parse_args()
    bundle = args.b
    models_path = args.p
    mode = args.mode
    verify(bundle, models_path, mode)
