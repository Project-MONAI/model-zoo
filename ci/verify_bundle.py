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
import shutil
import sys

import torch
from bundle_custom_data import (
    exclude_download_large_file_list,
    exclude_verify_preferred_files_list,
    exclude_verify_shape_list,
    exclude_verify_torchscript_list,
)
from monai.bundle import ckpt_export, verify_metadata, verify_net_in_out
from monai.bundle.config_parser import ConfigParser
from monai.utils.module import optional_import
from utils import download_large_files, get_json_dict

create_workflow, has_create_workflow = optional_import("monai.bundle", name="create_workflow")

# files that must be included in a bundle
necessary_files_list = ["configs/metadata.json", "LICENSE"]
# files that are preferred to be included in a bundle
preferred_files_list = ["models/model.pt", "configs/inference.json"]


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
    if bundle == "pediatric_abdominal_ct_segmentation":
        # skip test for this bundle's ts file
        return "dynunet_FT.pt", None
    if bundle == "brain_image_synthesis_latent_diffusion":
        return "autoencoder.pt", "model.pt"
    if bundle == "cxr_image_synthesis_latent_diffusion_model":
        return "autoencoder.pt", None
    return "model.pt", "model.ts"


def _get_net_id(bundle: str):
    # TODO: this function is temporarily used. It should be replaced by detailed config tests.
    if bundle == "brats_mri_generative_diffusion":
        return "autoencoder_def"
    if bundle == "brats_mri_axial_slices_generative_diffusion":
        return "autoencoder_def"
    return "network_def"


def _produce_fake_weights(config_file: str, meta_file: str, bundle: str, device: str):
    """
    This function is used to produce fake weights for a network.

    """
    parser = ConfigParser()
    parser.read_config(f=config_file)
    parser.read_meta(f=meta_file)
    net_id = _get_net_id(bundle)
    output_file = _get_weights_names(bundle)[0]
    network = parser.get_parsed_content(net_id).to(device)
    torch.save(network.state_dict(), output_file)


def _find_license_file(bundle_path: str):
    """
    Searches for a LICENSE file or a file starting with LICENSE. with any extension in the bundle path.
    Returns the path to the license file if found, otherwise None.
    """
    # Check for a file exactly named LICENSE
    license_exact_path = os.path.join(bundle_path, "LICENSE")
    if os.path.exists(license_exact_path):
        return license_exact_path

    # Return None if no license file is found
    return None


def verify_bundle_directory(models_path: str, bundle_name: str, mode: str):
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
    if bundle_name in exclude_download_large_file_list and mode == "min":
        print(f"skip downloading large files for bundle: {bundle_name}.")
    else:
        large_file_name = _find_bundle_file(bundle_path, "large_files")
        if large_file_name is not None:
            try:
                download_large_files(bundle_path=bundle_path, large_file_name=large_file_name)
            except Exception as e:
                raise ValueError(f"Download large files in {bundle_path} error.") from e

    # verify necessary files are included
    for file in necessary_files_list:
        if file == "LICENSE":
            # check if LICENSE file exists
            license_file_path = _find_license_file(bundle_path)
            if license_file_path is None:
                raise ValueError(f"necessary file {file} is not existing.")
        else:
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
    if ts_name is not None:
        ts_model_path = os.path.join(bundle_path, "models", ts_name)
        if os.path.exists(ts_model_path):
            torch.jit.load(ts_model_path)
            print("Provided TorchScript module is verified correctly.")


def get_app_properties(app: str, version: str):
    """
    This function is used to get the properties file of the app.

    """
    # dir structure: ./app/
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if "ci" in cur_dir:
        cur_dir = os.path.dirname(cur_dir)
    for root, _dirs, files in os.walk(os.path.join(cur_dir, app)):
        if "bundle_properties.py" in files and os.path.isfile(os.path.join(root, "bundle_properties.py")):
            print(root)
            return os.path.join(root, "bundle_properties.py")
    else:
        return None


def check_properties(**kwargs):
    """
    This function is used to check the properties of the workflow.
    """
    app_properties_path = kwargs.get("properties_path", "")
    kwargs.pop("properties_path", None)
    print(kwargs)

    if app_properties_path is not None and os.path.isfile(app_properties_path):
        shutil.copy(app_properties_path, "ci/bundle_properties.py")
        from bundle_properties import InferProperties, MetaProperties

        workflow = create_workflow(**kwargs)
        workflow.properties = {**MetaProperties, **InferProperties}
        check_result = workflow.check_properties()
        if check_result is not None and len(check_result) > 0:
            raise ValueError(
                f"check properties for workflow failed: {check_result}, app_properties_path: {app_properties_path}"
            )
        else:
            print(f"check properties for workflow successfully {check_result}.")


def verify_bundle_properties(model_path: str, bundle: str):
    """
    This function is used to verify the bundle properties.
    If a bundle supports multiple apps, the properties of each app should be checked.

    """
    bundle_path = os.path.join(model_path, bundle)
    meta_file = os.path.join(bundle_path, "configs/metadata.json")
    metadata = get_json_dict(meta_file)
    # Since lack of data, train and evaluate config properties can checked in unit tests
    # In this file, only inference config properties will be checked
    for workflow_type in ["inference"]:
        config_name = _find_bundle_file(os.path.join(bundle_path, "configs"), workflow_type)
        if config_name is not None:
            config_file = os.path.join(bundle_path, f"configs/{config_name}")
            meta_file = os.path.join(bundle_path, "configs/metadata.json")
            check_property_args = {
                "workflow_type": workflow_type,
                "bundle_root": bundle_path,
                "config_file": config_file,
                "logging_file": os.path.join(bundle_path, "configs/logging.conf"),
                "meta_file": meta_file,
            }
            if "supported_apps" in metadata:
                supported_apps = metadata["supported_apps"]
                all_properties = []
                for app, version in supported_apps.items():
                    properties_path = get_app_properties(app, version)
                    if properties_path is not None:
                        all_properties.append(properties_path)
                all_properties = list(set(all_properties))
                print("all properties: ", all_properties)
                for properties_path in all_properties:
                    check_property_args["properties_path"] = properties_path
                    check_properties(**check_property_args)
                    print("successfully checked properties.")
            else:
                # skip property check if supported_apps is not provided
                pass


def verify(bundle, models_path="models", mode="full"):
    print(f"start verifying {bundle}:")
    # add bundle path to ensure custom code can be used
    sys.path = [os.path.join(models_path, bundle)] + sys.path
    # verify bundle directory
    verify_bundle_directory(models_path, bundle, mode)
    print("directory is verified correctly.")
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

    # verify bundle properties
    verify_bundle_properties(models_path, bundle)
    print("properties are verified correctly.")

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
