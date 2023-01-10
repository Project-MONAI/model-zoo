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
# keys that must be included in inference config
infer_keys_list = ["bundle_root", "device", "network_def", "inferer"]
# keys that must be included in train config
train_keys_list = ["bundle_root", "device", "dataset_dir"]


def _find_bundle_file(root_dir: str, file: str, suffix=("json", "yaml", "yml")):
    # find bundle file with possible suffix
    file_name = None
    for name in suffix:
        full_name = f"{file}.{name}"
        if full_name in os.listdir(root_dir):
            file_name = full_name

    return file_name


def _check_missing_keys(file_name: str, bundle_path: str, keys_list: List):
    config = ConfigParser.load_config_file(os.path.join(bundle_path, "configs", file_name))
    missing_keys = []
    for key in keys_list:
        if key not in config:
            missing_keys.append(key)

    if len(missing_keys) > 0:
        raise ValueError(f"missing key(s): {str(missing_keys)} in {file_name}.")

    return config


def _check_main_section_necessary_key(necessary_key: str, config: dict, main_section: str = "train"):
    # `necessary_key` must be in `main_section`
    if necessary_key not in config[main_section]:
        raise ValueError(f"'{necessary_key}' is not existing in '{main_section}'.")


def _check_sub_section_necessary_key(
    necessary_key: str, config: dict, main_section: str = "train", sub_section: str = "trainer"
):
    # `necessary_key` must be in `sub_section`
    if necessary_key not in config[main_section][sub_section]:
        raise ValueError(f"'{necessary_key}' is not existing in '{main_section}#{sub_section}'.")


def _check_main_section_optional_key(
    arg_name: str, necessary_key: str, config: dict, main_section: str = "train", sub_section: str = "trainer"
):
    # if `arg_name` is in `sub_section`, its value must be `necessary_key`
    if arg_name in config[main_section][sub_section]:
        if necessary_key not in config[main_section]:
            actual_key = str(config[main_section][sub_section][arg_name]).split("#")[-1]
            raise ValueError(f"'{main_section}' should have '{necessary_key}', got '{actual_key}'.")


def _check_validation_handler(var_name: str, config: dict):
    if "handlers" in config:
        for handler in config["handlers"]:
            if handler["_target_"] == "ValidationHandler":
                interval_name = str(handler["interval"]).split("@")[-1]
                if not interval_name == var_name:
                    raise ValueError(
                        f"variable '{var_name}' should be defined for 'ValidationHandler', got '{interval_name}'."
                    )


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
    This function is used to verify if necessary keys are included in config files.

    """
    bundle_path = os.path.join(models_path, bundle_name)

    # verify inference config (if exists)
    inference_file_name = _find_bundle_file(os.path.join(bundle_path, "configs"), "inference")
    if inference_file_name is not None:
        _ = _check_missing_keys(file_name=inference_file_name, bundle_path=bundle_path, keys_list=infer_keys_list)

    # verify train config (if exists)
    train_file_name = _find_bundle_file(os.path.join(bundle_path, "configs"), "train")
    if train_file_name is not None:
        train_config = _check_missing_keys(
            file_name=train_file_name, bundle_path=bundle_path, keys_list=train_keys_list
        )

        if "train" in train_config:
            _check_main_section_necessary_key(necessary_key="trainer", config=train_config)
            _check_main_section_necessary_key(necessary_key="dataset", config=train_config)
            _check_main_section_necessary_key(necessary_key="handlers", config=train_config)
            _check_sub_section_necessary_key(necessary_key="max_epochs", config=train_config, sub_section="trainer")
            _check_sub_section_necessary_key(necessary_key="data", config=train_config, sub_section="dataset")
            _check_main_section_optional_key(
                arg_name="postprocessing", necessary_key="postprocessing", config=train_config, sub_section="trainer"
            )
            _check_main_section_optional_key(
                arg_name="transform", necessary_key="preprocessing", config=train_config, sub_section="dataset"
            )
            _check_main_section_optional_key(
                arg_name="key_train_metric", necessary_key="key_metric", config=train_config, sub_section="trainer"
            )
            # special requirements: if "ValidationHandler" in "handlers", key "val_interval" should be defined.
            _check_validation_handler(var_name="val_interval", config=train_config["train"])
        if "validate" in train_config:
            _check_main_section_necessary_key(necessary_key="evaluator", config=train_config, main_section="validate")
            _check_main_section_necessary_key(necessary_key="dataset", config=train_config, main_section="validate")
            _check_main_section_necessary_key(necessary_key="handlers", config=train_config, main_section="validate")
            _check_sub_section_necessary_key(
                necessary_key="data", config=train_config, main_section="validate", sub_section="dataset"
            )
            _check_main_section_optional_key(
                arg_name="postprocessing",
                necessary_key="postprocessing",
                config=train_config,
                main_section="validate",
                sub_section="evaluator",
            )
            _check_main_section_optional_key(
                arg_name="transform",
                necessary_key="preprocessing",
                config=train_config,
                main_section="validate",
                sub_section="dataset",
            )
            _check_main_section_optional_key(
                arg_name="inferer",
                necessary_key="inferer",
                config=train_config,
                main_section="validate",
                sub_section="evaluator",
            )
            _check_main_section_optional_key(
                arg_name="key_val_metric",
                necessary_key="key_metric",
                config=train_config,
                main_section="validate",
                sub_section="evaluator",
            )


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


def verify_torchscript(bundle_path: str, net_id: str, config_file: str):
    """
    This function is used to verify if the checkpoint is able to torchscript, and
    if "models/model.ts" is provided, it will be checked if is able to be loaded
    successfully.

    """
    ckpt_export(
        net_id=net_id,
        filepath=os.path.join(bundle_path, "models/verify_model.ts"),
        ckpt_file=os.path.join(bundle_path, "models/model.pt"),
        meta_file=os.path.join(bundle_path, "configs/metadata.json"),
        config_file=os.path.join(bundle_path, config_file),
        bundle_root=bundle_path,
    )
    print("export weights into TorchScript module successfully.")

    ts_model_path = os.path.join(bundle_path, "models/model.ts")
    if os.path.exists(ts_model_path):
        _ = torch.jit.load(ts_model_path)
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
    net_id, inference_file_name = "network_def", _find_bundle_file(os.path.join(bundle_path, "configs"), "inference")
    config_file = os.path.join("configs", inference_file_name)

    if bundle in exclude_verify_shape_list:
        print(f"skip verifying the data shape of bundle: {bundle}.")
    else:
        verify_data_shape(bundle_path, net_id, config_file)
        print("data shape is verified correctly.")

    if bundle in exclude_verify_torchscript_list:
        print(f"bundle: {bundle} does not support torchscript, skip verifying.")
    else:
        verify_torchscript(bundle_path, net_id, config_file)


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
