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

from bundle_custom_data import custom_net_config_dict, exclude_verify_shape_list, exclude_verify_torchscript_list
from monai.bundle import ckpt_export, verify_metadata, verify_net_in_out
from utils import download_large_files, get_changed_bundle_list, get_json_dict


def verify_bundle_directory(models_path: str, bundle_name: str):
    """
    According to [MONAI Bundle Specification](https://docs.monai.io/en/latest/mb_specification.html),
    "configs/metadata.json" is required within the bundle root directory.
    This function is used to verify this file. For bundles that contain the download links for large
    files, the links should be saved in "large_files.yml" (or .json, .yaml).
    All large files (if exist) will be downloaded before verification.

    """

    bundle_path = os.path.join(models_path, bundle_name)
    large_file_name = None
    for name in os.listdir(bundle_path):
        if name in ["large_files.yml", "large_files.yaml", "large_files.json"]:
            large_file_name = name

    if large_file_name is not None:
        try:
            download_large_files(bundle_path=bundle_path, large_file_name=large_file_name)
        except Exception as e:
            raise ValueError(f"Download large files in {bundle_path} error.") from e

    metadata_path = os.path.join(bundle_path, "configs/metadata.json")
    if not os.path.exists(metadata_path):
        raise ValueError(f"metadata path: {metadata_path} is not existing.")


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
    bundle_zip_name = f"{bundle_name}_v{latest_version}.zip"
    if bundle_zip_name in model_info.keys():
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


def verify_export_torchscript(bundle_path: str, net_id: str, config_file: str):
    """
    This function is used to verify if the checkpoint is able to torchscript.

    """
    ckpt_export(
        net_id=net_id,
        filepath=os.path.join(bundle_path, "models/verify_model.ts"),
        ckpt_file=os.path.join(bundle_path, "models/model.pt"),
        meta_file=os.path.join(bundle_path, "configs/metadata.json"),
        config_file=os.path.join(bundle_path, config_file),
        bundle_root=bundle_path,
    )


def get_net_id_config_name(bundle_name: str):
    """
    Return values of arguments net_id and config_file.
    """
    net_id = "network_def"
    config_file = "configs/inference.json"
    if bundle_name not in custom_net_config_dict.keys():
        return net_id, config_file

    name_dict = custom_net_config_dict[bundle_name]
    if "net_id" in name_dict.keys():
        net_id = name_dict["net_id"]
    if "config_file" in name_dict.keys():
        config_file = name_dict["config_file"]

    return net_id, config_file


def main(changed_dirs):
    """
    main function to process all changed files. It will do the following steps:

    1. according to changed directories, get changed bundles.
    2. verify each bundle.

    """
    bundle_list = get_changed_bundle_list(changed_dirs)
    models_path = "models"

    if len(bundle_list) > 0:
        for bundle in bundle_list:
            # verify bundle directory
            verify_bundle_directory(models_path, bundle)
            # verify version, changelog
            verify_version_changes(models_path, bundle)
            # verify metadata format and data
            bundle_path = os.path.join(models_path, bundle)
            verify_metadata_format(bundle_path)

            # The following are optional tests
            net_id, config_file = get_net_id_config_name(bundle)

            if bundle in exclude_verify_shape_list:
                print(f"skip verifying the data shape of bundle: {bundle}.")
            else:
                verify_data_shape(bundle_path, net_id, config_file)
            if bundle in exclude_verify_torchscript_list:
                print(f"bundle: {bundle} does not support torchscript, skip verifying.")
            else:
                verify_export_torchscript(bundle_path, net_id, config_file)
    else:
        print(f"all changed files: {changed_dirs} are not related to any existing bundles, skip verifying.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--f", type=str, help="changed files.")
    args = parser.parse_args()
    changed_dirs = args.f.splitlines()
    main(changed_dirs)
