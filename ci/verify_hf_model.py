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

from monai.bundle import verify_metadata
from utils import get_json_dict


def verify_hf_model_directory(models_path: str, model_name: str):
    """
    Required files:
    - README.md
    - LICENSE
    - metadata.json

    """

    necessary_files_list = ["README.md", "LICENSE", "metadata.json"]

    model_path = os.path.join(models_path, model_name)
    # verify necessary files are included
    for file in necessary_files_list:
        if not os.path.exists(os.path.join(model_path, file)):
            raise ValueError(f"necessary file {file} is not existing.")


def verify_version_changes(models_path: str, model_name: str):
    """
    This function is used to verify if "version" and "changelog" are correct in "metadata.json".
    In addition, if changing an existing hf model, a new version number should be provided.

    """

    model_path = os.path.join(models_path, model_name)

    meta_file_path = os.path.join(model_path, "metadata.json")
    metadata = get_json_dict(meta_file_path)
    if "version" not in metadata:
        raise ValueError(f"'version' is missing in metadata.json of hf model: {model_name}.")
    if "changelog" not in metadata:
        raise ValueError(f"'changelog' is missing in metadata.json of hf model: {model_name}.")

    # version number should be in changelog
    latest_version = metadata["version"]
    if latest_version not in metadata["changelog"].keys():
        raise ValueError(
            f"version number: {latest_version} is missing in 'changelog' in metadata.json of hf model: {model_name}."
        )


def verify_metadata_format(model_path: str):
    """
    This function is used to verify the metadata format.

    """
    verify_metadata(
        meta_file=os.path.join(model_path, "metadata.json"), filepath=os.path.join(model_path, "eval/schema.json")
    )


def verify(model_name, models_path="hf_models", mode="full"):
    print(f"start verifying {model_name}:")
    # add bundle path to ensure custom code can be used
    sys.path = [os.path.join(models_path, model_name)] + sys.path
    # verify bundle directory
    verify_hf_model_directory(models_path, model_name)
    print("directory is verified correctly.")
    if mode != "regular":
        # verify version, changelog
        verify_version_changes(models_path, model_name)
        print("version and changelog are verified correctly.")
    # verify metadata format and data
    model_path = os.path.join(models_path, model_name)
    verify_metadata_format(model_path)
    print("metadata format is verified correctly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-b", "--b", type=str, help="model name.")
    parser.add_argument("-p", "--p", type=str, default="hf_models", help="models path.")
    parser.add_argument("-m", "--mode", type=str, default="full", help="verify model mode (full/min).")
    args = parser.parse_args()
    model_name = args.b
    models_path = args.p
    mode = args.mode
    verify(model_name, models_path, mode)
