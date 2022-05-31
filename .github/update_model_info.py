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
import subprocess
import tempfile
from utils import (
    get_sub_folders,
    get_json_dict,
    get_hash_func,
    get_changed_bundle_list,
    download_large_files,
)
from typing import List
import json


def update_model_info(bundle_list: List, models_path: str = "models", model_info_file: str = "model_info.json"):
    model_info_path = os.path.join(models_path, model_info_file)
    model_info = get_json_dict(model_info_path)
    hash_func = get_hash_func(hash_type="sha1")
    # make a temp dir to store compressed bundles
    temp_dir = tempfile.mkdtemp()
    # temp_dir = "test_tmp"
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)

    for bundle_name in bundle_list:
        if bundle_name not in model_info.keys():
            model_info[bundle_name] = {"version": "", "checksum": "", "source": ""}
        # copy the bundle into temp dir for further compress and upload
        bundle_path = os.path.join(temp_dir, bundle_name)
        shutil.copytree(os.path.join(models_path, bundle_name), bundle_path)
        # need to check if metadata.json is existing in the bundle when submit a PR.
        # get version from metadata
        bundle_metadata_path = os.path.join(bundle_path, "configs/metadata.json")
        metadata = get_json_dict(bundle_metadata_path)
        latest_version = metadata["version"]

        # download large files if exists
        for large_file_type in [".yml", ".yaml", ".json"]:
            large_file_name = "large_file" + large_file_type
            large_file_path = os.path.join(bundle_path, large_file_name)
            if os.path.exists(large_file_path):
                download_large_files(bundle_path=bundle_path, large_file_name=large_file_name)
                # remove the large file config
                os.remove(large_file_path)

        # compress bundle
        bundle_zip_name = f"{bundle_name}_v{latest_version}.zip"
        zipfile_path = os.path.join(temp_dir, bundle_zip_name)
        compress_bundle(root_path=temp_dir, bundle_name=bundle_name, bundle_zip_name=bundle_zip_name)
        # get checksum
        checksum = get_checksum(dst_path=zipfile_path, hash_func=hash_func)
        # upload new zip
        source = upload_bundle(bundle_zip_file_path=zipfile_path, bundle_zip_filename=bundle_zip_name)
        # update model_info
        model_info[bundle_name]["checksum"] = checksum
        model_info[bundle_name]["version"] = latest_version
        model_info[bundle_name]["source"] = source

    print(model_info)
    save_model_info(model_info, model_info_path)

    shutil.rmtree(temp_dir)


def save_model_info(model_info_dict, model_info_path: str):
    with open(model_info_path, "w") as f:
        json.dump(model_info_dict, f)

    # merged_pr_num = os.environ["PR_NUMBER"]
    # email = os.environ["email"]
    # username = os.environ["username"]

    # branch_name = f"{merged_pr_num}-auto-update-model-info"
    # create_push_cmd = f"git checkout -b {branch_name}; git push --set-upstream origin {branch_name}"

    # git_config = f"git config user.email {email}; git config user.name {username}"
    # commit_message = "git commit -m 'auto update model_info'"
    # full_cmd = f"{git_config}; git add {model_info_path}; {commit_message}; {create_push_cmd}"

    # subprocess.run(full_cmd, shell=True)


def compress_bundle(root_path: str, bundle_name: str, bundle_zip_name: str):

    touch_cmd = f"find {bundle_name} -exec touch -t 202205300000 " + "{} +"
    zip_cmd = f"zip -rq -D -X -9 -A --compression-method deflate {bundle_zip_name} {bundle_name}"
    subprocess.call(f"{touch_cmd}; {zip_cmd}", shell=True, cwd=root_path)


def get_checksum(dst_path: str, hash_func):
    with open(dst_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def upload_bundle(
    bundle_zip_file_path: str,
    bundle_zip_filename: str,
    release_tag: str = "hosting_storage_v1",
    repo_name: str = "Project-MONAI/model-zoo",
):

    upload_command = f"gh release upload {release_tag} {bundle_zip_file_path} -R {repo_name}"
    subprocess.run(upload_command, shell=True)
    source = f"https://github.com/{repo_name}/releases/download/{release_tag}/{bundle_zip_filename}"

    return source


def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-b", "--bundle", type=str, help="bundle names to be updated.")
    args = parser.parse_args()
    bundle_list = [args.bundle]

    update_model_info(bundle_list=bundle_list)


if __name__ == "__main__":
    main()
