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
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

from monai.utils import look_up_option

SUPPORTED_HASH_TYPES = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256, "sha512": hashlib.sha512}


def get_sub_folders(root_dir: str):
    return [f.name for f in os.scandir(root_dir) if f.is_dir()]


def get_json_dict(json_dict_path: str):
    with open(json_dict_path, "r") as f:
        json_dict = json.load(f)
    return json_dict


def get_hash_func(hash_type: str = "sha1"):
    actual_hash_func = look_up_option(hash_type.lower(), SUPPORTED_HASH_TYPES)
    return actual_hash_func()


def update_model_info(models_path: str, model_info_file: str = "model_info.json"):
    model_info_path = os.path.join(models_path, model_info_file)
    model_info = get_json_dict(model_info_path)
    hash_func = get_hash_func()
    # make a temp dir to store compressed bundles
    temp_dir = tempfile.mkdtemp()
    ischanged = False
    task_list = get_sub_folders(models_path)
    for task in task_list:
        if task not in model_info.keys():
            model_info[task] = {}
        task_path = os.path.join(models_path, task)
        task_bundle_list = get_sub_folders(task_path)
        for bundle in task_bundle_list:
            if bundle not in model_info[task].keys():
                model_info[task][bundle] = {"version": "", "checksum": "", "source": ""}
            bundle_path = os.path.join(task_path, bundle)
            bundle_metadata_path = os.path.join(bundle_path, "configs/metadata.json")
            if os.path.exists(bundle_metadata_path):
                metadata = get_json_dict(bundle_metadata_path)
                # get version number from metadata
                latest_version = metadata["version"]
                # compress bundle
                bundle_zip_name = f"{bundle}_v{latest_version}.tar"
                compress_bundle(root_path=task_path, bundle_name=bundle, bundle_zip_name=bundle_zip_name)
                dst_path = os.path.join(temp_dir, bundle_zip_name)
                shutil.move(os.path.join(task_path, bundle_zip_name), dst_path)
                # get actual checksum
                actual_checksum = get_checksum(dst_path=dst_path, hash_func=hash_func)

                # check consistency
                if model_info[task][bundle]["checksum"] != actual_checksum:
                    # upload new zip
                    new_source = upload_bundle(dst_path, bundle_zip_name)
                    # update model_info
                    model_info[task][bundle]["checksum"] = actual_checksum
                    model_info[task][bundle]["version"] = latest_version
                    model_info[task][bundle]["source"] = new_source
                    ischanged = True

    if ischanged is True:
        push_model_info(model_info, model_info_path)

    shutil.rmtree(temp_dir)


def push_model_info(model_info_dict, model_info_path: str):
    with open(model_info_path, "w") as f:
        json.dump(model_info_dict, f)

    merged_pr_num = os.environ["PR_NUMBER"]
    email = os.environ["email"]
    username = os.environ["username"]

    branch_name = f"{merged_pr_num}-auto-update-model-info"
    create_push_cmd = f"git checkout -b {branch_name}; git push --set-upstream origin {branch_name}"

    git_config = f"git config user.email {email}; git config user.name {username}"
    commit_message = "git commit -m 'auto update model_info'"
    full_cmd = f"{git_config}; git add {model_info_path}; {commit_message}; {create_push_cmd}"

    subprocess.run(full_cmd, shell=True)


def compress_bundle(root_path: str, bundle_name: str, bundle_zip_name: str):

    deterministic_cmd = (
        "--sort=name --owner=0 --group=0 --mode='go-rwx,u-rw' --numeric-owner --mtime='UTC 2022-06-01' -c"
    )
    subprocess.call(f"tar {deterministic_cmd} {bundle_name} | gzip -n > {bundle_zip_name}", shell=True, cwd=root_path)


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

    parser.add_argument("-p", "--path", default="models", help="path of models folder.")
    args = parser.parse_args()
    update_model_info(models_path=args.path)


if __name__ == "__main__":
    main()
