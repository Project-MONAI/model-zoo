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

from utils_deparate import (
    compress_bundle,
    create_pull_request,
    download_large_files,
    get_changed_bundle_list,
    get_checksum,
    get_hash_func,
    get_json_dict,
    push_new_model_info_branch,
    save_model_info,
    upload_bundle,
)


def update_model_info(
    bundle_name: str, temp_dir: str, models_path: str = "models", model_info_file: str = "model_info.json"
):
    """
    For a changed model (bundle), this function is used to do the following steps in order to update it:

    1. download large files (if having the corresponding config file) into the copy.
    2. compress the copy.
    3. upload a compressed copy.
    4. update `model_info_file`.

    Returns:
        a 2-tuple.
        If update successful, the form is (True,"").
        If update failed, the form is (False, "error reason")
    """
    temp_path = os.path.join(temp_dir, bundle_name)
    shutil.copytree(os.path.join(models_path, bundle_name), temp_path)
    # step 1
    try:
        for large_file_type in [".yml", ".yaml", ".json"]:
            large_file_name = "large_files" + large_file_type
            large_file_path = os.path.join(temp_path, large_file_name)
            if os.path.exists(large_file_path):
                download_large_files(bundle_path=temp_path, large_file_name=large_file_name)
                # remove the large file config
                os.remove(large_file_path)
    except Exception as e:
        return (False, f"Download large files error: {e}")

    # step 2
    bundle_metadata_path = os.path.join(temp_path, "configs/metadata.json")
    metadata = get_json_dict(bundle_metadata_path)
    latest_version = metadata["version"]
    bundle_zip_name = f"{bundle_name}_v{latest_version}.zip"
    bundle_name_with_version = f"{bundle_name}_v{latest_version}"
    zipfile_path = os.path.join(temp_dir, bundle_zip_name)
    try:
        compress_bundle(root_path=temp_dir, bundle_name=bundle_name, bundle_zip_name=bundle_zip_name)
    except Exception as e:
        return (False, f"Compress bundle error: {e}")

    hash_func = get_hash_func(hash_type="sha1")
    checksum = get_checksum(dst_path=zipfile_path, hash_func=hash_func)

    # step 3
    try:
        source = upload_bundle(bundle_zip_file_path=zipfile_path, bundle_zip_filename=bundle_zip_name)
    except Exception as e:
        return (False, f"Upload bundle error: {e}")

    # step 4
    model_info_path = os.path.join(models_path, model_info_file)
    model_info = get_json_dict(model_info_path)

    if bundle_name_with_version not in model_info.keys():
        model_info[bundle_name_with_version] = {"checksum": "", "source": ""}

    model_info[bundle_name_with_version]["checksum"] = checksum
    model_info[bundle_name_with_version]["source"] = source

    save_model_info(model_info, model_info_path)
    return (True, "update successful")


def main(changed_dirs):
    """
    main function to process all changed files. It will do the following steps:

    1. according to changed directories, get changed bundles.
    2. update each bundle.
    3. according to the update results, push changed model_info_file if needed.

    """
    bundle_list = get_changed_bundle_list(changed_dirs)
    models_path = "models"
    model_info_file = "model_info.json"
    if len(bundle_list) > 0:
        for bundle in bundle_list:
            # create a temporary copy of the bundle for further processing
            temp_dir = tempfile.mkdtemp()
            update_state, msg = update_model_info(
                bundle_name=bundle, temp_dir=temp_dir, models_path=models_path, model_info_file=model_info_file
            )
            shutil.rmtree(temp_dir)

            if update_state is True:
                print(f"update bundle: {bundle} successful.")
            else:
                raise AssertionError(f"update bundle: {bundle} failed. {msg}")

        # push a new branch that contains the updated model_info.json
        branch_name = push_new_model_info_branch(model_info_path=os.path.join(models_path, model_info_file))
        create_pull_request(branch_name)
    else:
        print(f"all changed files: {changed_dirs} are not related to any existing bundles, skip updating.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--f", type=str, help="changed files.")
    args = parser.parse_args()
    changed_dirs = args.f.splitlines()
    main(changed_dirs)
