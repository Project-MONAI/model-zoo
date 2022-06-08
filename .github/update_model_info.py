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
import tempfile
import warnings

from utils import (
    compress_bundle,
    download_large_files,
    get_changed_bundle_list,
    get_checksum,
    get_hash_func,
    get_json_dict,
    save_model_info,
    upload_bundle,
)


def update_model_info(bundle_name: str, models_path: str = "models", model_info_file: str = "model_info.json"):
    """
    For a changed model (bundle), this function is used to do the following steps in order to update it:

    1. check if bundle folder exists.
    2. create a temporary copy of the bundle.
    3. download large files (if having the corresponding config file) into the copy.
    4. compress the copy.
    5. upload a compressed copy.
    6. update `model_info_file`.

    Returns:
        a 2-tuple.
        If update successful, the form is (True,"").
        If update failed, the form is (False, "error reason")
    """

    # step 1
    bundle_path = os.path.join(models_path, bundle_name)
    if not os.path.exists(bundle_path):
        warnings.warn(f"bundle path: {bundle_path} not exists, skip update.")
        return (False, "bundle path not exist")

    # temp_dir = "test_tmp"
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)

    # step 2
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, bundle_name)
    shutil.copytree(bundle_path, temp_path)

    # step 3
    try:
        for large_file_type in [".yml", ".yaml", ".json"]:
            large_file_name = "large_files" + large_file_type
            large_file_path = os.path.join(temp_path, large_file_name)
            if os.path.exists(large_file_path):
                download_large_files(bundle_path=temp_path, large_file_name=large_file_name)
                # remove the large file config
                os.remove(large_file_path)
    except Exception as e:
        shutil.rmtree(temp_dir)
        return (False, f"download large files error: {e}")

    # step 4
    bundle_metadata_path = os.path.join(temp_path, "configs/metadata.json")
    metadata = get_json_dict(bundle_metadata_path)
    latest_version = metadata["version"]
    bundle_zip_name = f"{bundle_name}_v{latest_version}.zip"
    zipfile_path = os.path.join(temp_dir, bundle_zip_name)
    try:
        compress_bundle(root_path=temp_dir, bundle_name=bundle_name, bundle_zip_name=bundle_zip_name)
    except Exception as e:
        shutil.rmtree(temp_dir)
        return (False, f"compress bundle error: {e}")

    hash_func = get_hash_func(hash_type="sha1")
    checksum = get_checksum(dst_path=zipfile_path, hash_func=hash_func)

    # step 5
    try:
        source = upload_bundle(bundle_zip_file_path=zipfile_path, bundle_zip_filename=bundle_zip_name)
    except Exception as e:
        shutil.rmtree(temp_dir)
        return (False, f"upload bundle error: {e}")

    # step 6
    model_info_path = os.path.join(models_path, model_info_file)
    model_info = get_json_dict(model_info_path)

    if bundle_name not in model_info.keys():
        model_info[bundle_name] = {"version": "", "checksum": "", "source": ""}

    model_info[bundle_name]["checksum"] = checksum
    model_info[bundle_name]["version"] = latest_version
    model_info[bundle_name]["source"] = source

    save_model_info(model_info, model_info_path)

    shutil.rmtree(temp_dir)
    return (True, "update successful")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--f", type=str, help="changed files.")
    args = parser.parse_args()
    changed_dirs = args.f.splitlines()
    bundle_list = get_changed_bundle_list(changed_dirs)
    if len(bundle_list) > 0:
        for bundle in bundle_list:
            update_state, msg = update_model_info(bundle_name=bundle)
            if update_state is True:
                print(f"update bundle: {bundle} successful.")
            else:
                print(f"update bundle: {bundle} failed, error: {msg}")


if __name__ == "__main__":
    main()
