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

import torch
from monai.bundle import ckpt_export, verify_metadata, verify_net_in_out
from utils import download_large_files, get_changed_bundle_list


def verify_download_large_files(bundle_path: str):
    """
    This function is used to verify the `large_files` file (if exists) of the input bundle path.
    Wrong download link or checksum will raise an error.
    A successful verification will also keep the downloaded large files for other verifications.

    """
    # verify download
    try:
        for large_file_type in [".yml", ".yaml", ".json"]:
            large_file_name = "large_files" + large_file_type
            large_file_path = os.path.join(bundle_path, large_file_name)
            if os.path.exists(large_file_path):
                download_large_files(bundle_path=bundle_path, large_file_name=large_file_name)
    except Exception as e:
        raise ValueError(f"Download large files in {bundle_path} error") from e


def verify_metadata_format(bundle_path: str):
    """
    This function is used to verify the metadata format.

    """
    verify_metadata(
        meta_file=os.path.join(bundle_path, "configs/metadata.json"),
        filepath=os.path.join(bundle_path, "eval/schema.json"),
    )


def verify_data_shape(inference_file: str, bundle_path: str):
    """
    This function is used to verify the data shape of network.

    """
    verify_net_in_out(
        net_id="network_def",
        meta_file=os.path.join(bundle_path, "configs/metadata.json"),
        config_file=os.path.join(bundle_path, f"configs/{inference_file}"),
        bundle_root=bundle_path,
    )


def verify_export_torchscript(inference_file: str, bundle_path: str):
    """
    This function is used to verify if the checkpoint is able to torchscript.

    """
    ckpt_export(
        net_id="network_def",
        filepath=os.path.join(bundle_path, "models/verify_model.ts"),
        ckpt_file=os.path.join(bundle_path, "models/model.pt"),
        meta_file=os.path.join(bundle_path, "configs/metadata.json"),
        config_file=os.path.join(bundle_path, f"configs/{inference_file}"),
        bundle_root=bundle_path,
    )


def main(changed_dirs):
    """
    main function to process all changed files. It will do the following steps:

    1. according to changed directories, get changed bundles.
    2. verify each bundle on download large files, metadata format, data shape of network
        and torchscript export (if supports).

    """
    bundle_list = get_changed_bundle_list(changed_dirs)
    models_path = "models"

    if len(bundle_list) > 0:
        for bundle in bundle_list:
            bundle_path = os.path.join(models_path, bundle)

            # verify download
            verify_download_large_files(bundle_path)
            # verify metadata format and data
            verify_metadata_format(bundle_path)
            # verify data shape of network
            config_file_list = os.listdir(os.path.join(bundle_path, "configs"))
            inference_file = None
            for f in config_file_list:
                if "inference" in f:
                    inference_file = f
                    break
            if inference_file is None:
                raise ValueError("inference config file is missing.")
            verify_data_shape(inference_file, bundle_path)
            # verify export torchscript, only use when the device has gpu
            if torch.cuda.is_available() is True:
                if os.path.isfile(
                    os.path.join(bundle_path, "models/model.ts") and os.path.join(bundle_path, "models/model.pt")
                ):
                    verify_export_torchscript(inference_file, bundle_path)
                else:
                    print(f"bundle: {bundle} does not support torchscript, skip verifying.")

    else:
        print(f"all changed files: {changed_dirs} are not related to any existing bundles, skip verifying.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--f", type=str, help="changed files.")
    args = parser.parse_args()
    changed_dirs = args.f.splitlines()
    main(changed_dirs)
