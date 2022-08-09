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


import hashlib
import json
import os
import subprocess
from typing import List

from monai.apps.utils import download_url
from monai.bundle.config_parser import ConfigParser
from monai.utils import look_up_option

SUPPORTED_HASH_TYPES = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256, "sha512": hashlib.sha512}


def get_sub_folders(root_dir: str):
    """
    This function is used to get all sub folders (as a list) within the root_dir.
    """
    sub_folder_list = [f.name for f in os.scandir(root_dir) if f.is_dir()]

    return sub_folder_list


def get_json_dict(json_dict_path: str):
    with open(json_dict_path, "r") as f:
        json_dict = json.load(f)

    return json_dict


def get_hash_func(hash_type: str = "sha1"):
    actual_hash_func = look_up_option(hash_type.lower(), SUPPORTED_HASH_TYPES)

    return actual_hash_func()


def get_changed_bundle_list(changed_dirs: List[str], root_path: str = "models"):
    """
    This function is used to return all bundle names that have changed files.
    If a bundle is totally removed, it will be ignored (since it not exists).

    """
    bundles = get_sub_folders(root_path)

    changed_bundle_list = []
    for sub_dir in changed_dirs:
        for bundle in bundles:
            bundle_path = os.path.join(root_path, bundle)
            if os.path.commonpath([bundle_path]) == os.path.commonpath([bundle_path, sub_dir]):
                changed_bundle_list.append(bundle)

    return list(set(changed_bundle_list))


def download_large_files(bundle_path: str, large_file_name: str = "large_file.yml"):
    parser = ConfigParser()
    parser.read_config(os.path.join(bundle_path, large_file_name))
    large_files_list = parser.get()["large_files"]
    for lf_data in large_files_list:
        lf_data["fuzzy"] = True
        if lf_data["hash_val"] == "":
            lf_data.pop("hash_val")
        if lf_data["hash_type"] == "":
            lf_data.pop("hash_type")
        lf_data["filepath"] = os.path.join(bundle_path, lf_data["path"])
        lf_data.pop("path")
        download_url(**lf_data)


def save_model_info(model_info_dict, model_info_path: str):
    with open(model_info_path, "w") as f:
        json.dump(model_info_dict, f)


def push_new_model_info_branch(model_info_path: str):
    merged_pr_num = os.environ["PR_NUMBER"]
    email = os.environ["email"]
    username = os.environ["username"]

    branch_name = f"{merged_pr_num}-auto-update-model-info"
    create_push_cmd = f"git checkout -b {branch_name}; git push --set-upstream origin {branch_name}"

    git_config = f"git config user.email {email}; git config user.name {username}"
    commit_message = "git commit -m 'auto update model_info'"
    full_cmd = f"{git_config}; git add {model_info_path}; {commit_message}; {create_push_cmd}"

    call_status = subprocess.run(full_cmd, shell=True)
    call_status.check_returncode()

    return branch_name


def create_pull_request(branch_name: str):
    create_command = f"gh pr create --fill --title 'auto update model_info' --base dev --head {branch_name}"
    call_status = subprocess.run(create_command, shell=True)
    call_status.check_returncode()


def compress_bundle(root_path: str, bundle_name: str, bundle_zip_name: str):

    touch_cmd = f"find {bundle_name} -exec touch -t 202205300000 " + "{} +"
    zip_cmd = f"zip -rq -D -X -9 -A --compression-method deflate {bundle_zip_name} {bundle_name}"
    subprocess.check_call(f"{touch_cmd}; {zip_cmd}", shell=True, cwd=root_path)


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
    call_status = subprocess.run(upload_command, shell=True)
    call_status.check_returncode()
    source = f"https://github.com/{repo_name}/releases/download/{release_tag}/{bundle_zip_filename}"

    return source
