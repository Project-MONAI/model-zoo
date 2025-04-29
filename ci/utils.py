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
import re
import shutil
from typing import List

from monai.apps.utils import download_url
from monai.bundle.config_parser import ConfigParser
from monai.utils import look_up_option, optional_import

huggingface_hub, _ = optional_import("huggingface_hub")
Github, _ = optional_import("github", name="Github")

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


def prepare_schema(bundle_list: List[str], root_path: str = "models", hf_model: bool = False):
    """
    This function is used to prepare schema for changed bundles.
    Due to Github's limitation (see: https://github.com/Project-MONAI/model-zoo/issues/111),
    to avoid repeated downloading, all distinct schemas will be downloaded first, and copy
    to changed bundle directories.

    """
    schema_dict = {}
    for bundle_name in bundle_list:
        bundle_path = os.path.join(root_path, bundle_name)
        if os.path.exists(bundle_path):
            if hf_model:
                meta_file_path = os.path.join(bundle_path, "metadata.json")
            else:
                meta_file_path = os.path.join(bundle_path, "configs/metadata.json")
            metadata = get_json_dict(meta_file_path)
            schema_url = metadata["schema"]
            schema_name = schema_url.split("/")[-1]

            if schema_url not in schema_dict.keys():
                schema_path = os.path.join(root_path, schema_name)
                download_url(url=schema_url, filepath=schema_path)
                schema_dict[schema_url] = schema_path
            os.makedirs(os.path.join(bundle_path, "eval"), exist_ok=True)
            shutil.copyfile(schema_dict[schema_url], os.path.join(bundle_path, "eval/schema.json"))
            print("prepared schema for: ", bundle_name)


def download_large_files(bundle_path: str, large_file_name: str = "large_file.yml"):
    parser = ConfigParser()
    parser.read_config(os.path.join(bundle_path, large_file_name))
    large_files_list = parser.get()["large_files"]
    for lf_data in large_files_list:
        lf_data["fuzzy"] = True
        if "hash_val" in lf_data and lf_data.get("hash_val", "") == "":
            lf_data.pop("hash_val")
        if "hash_type" in lf_data and lf_data.get("hash_type", "") == "":
            lf_data.pop("hash_type")
        lf_data["filepath"] = os.path.join(bundle_path, lf_data["path"])
        lf_data.pop("path")
        download_url(**lf_data)


def save_model_info(model_info_dict, model_info_path: str):
    with open(model_info_path, "w") as f:
        json.dump(model_info_dict, f, indent=4)


def get_latest_version(bundle_name: str, model_info_path: str):
    model_info_dict = get_json_dict(model_info_path)
    versions = []
    for k in model_info_dict.keys():
        if bundle_name in k:
            versions.append(k.split(f"{bundle_name}_v")[1])

    return sorted(versions)[-1]


def get_version_checksum(bundle_name: str, version: str, model_info_path: str):
    model_info_dict = get_json_dict(model_info_path)
    return model_info_dict[f"{bundle_name}_v{version}"]["checksum"]


def submit_pull_request(model_info_path: str):
    # set required info for a pull request
    branch_name = "auto-update-model-info"
    pr_title = "auto update model_info [skip ci]"
    pr_description = "This PR is automatically created to update model_info.json"
    commit_message = "auto update model_info"
    repo_file_path = "models/model_info.json"
    # authenticate with Github CLI
    github_token = os.environ["GITHUB_TOKEN"]
    repo_name = "Project-MONAI/model-zoo"
    g = Github(github_token)
    # create new branch
    repo = g.get_repo(repo_name)
    default_branch = repo.default_branch
    new_branch = repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=repo.get_branch(default_branch).commit.sha)
    # push changes
    model_info = get_json_dict(model_info_path)
    repo.update_file(
        path=repo_file_path,
        message=commit_message,
        content=json.dumps(model_info),
        sha=repo.get_contents(repo_file_path, ref=default_branch).sha,
        branch=new_branch.ref,
    )
    # create PR
    repo.create_pull(title=pr_title, body=pr_description, head=new_branch.ref, base=default_branch)


def split_bundle_name_version(bundle_name: str):
    pattern_version = re.compile(r"^(.+)\_v(\d.*)$")
    matched_result = pattern_version.match(bundle_name)
    if matched_result is not None:
        b_name, b_version = matched_result.groups()
        return b_name, b_version
    raise ValueError(f"{bundle_name} does not meet the naming format.")


def get_existing_bundle_list(model_info):
    all_bundle_names = []
    for k in model_info.keys():
        bundle_name, _ = split_bundle_name_version(k)
        if bundle_name not in all_bundle_names:
            all_bundle_names.append(bundle_name)
    return all_bundle_names


def create_bundle_to_huggingface(bundle_name: str, org_name: str):
    api = huggingface_hub.HfApi()
    try:
        _ = api.create_repo(repo_id=f"{org_name}/{bundle_name}", repo_type="model", private=False)
    except Exception as e:
        if "already created" in str(e):
            print(f"{org_name}/{bundle_name} already exists, skip creating.")
        else:
            raise e


def upload_version_to_huggingface(bundle_name: str, version: str, root_path: str, org_name: str):
    api = huggingface_hub.HfApi()

    try:
        # if no file is changed, will skip uploading automatically

        api.upload_folder(
            folder_path=os.path.join(root_path, bundle_name),
            repo_id=f"{org_name}/{bundle_name}",
            repo_type="model",
            commit_message=f"Upload {bundle_name} version {version}",
        )
    except Exception as e:
        print(f"Error uploading {bundle_name} to Hugging Face: {e}")
        raise e

    # tag version
    try:
        api.create_tag(
            repo_id=f"{org_name}/{bundle_name}",
            tag=version,
            repo_type="model",
            tag_message=f"tag {bundle_name} version {version}",
        )
    except Exception as e:
        if "Tag reference exists already" in str(e):
            print(f"Tag {version} already exists, skip creating.")
        else:
            print(f"Error tagging {bundle_name} version {version}: {e}")
            raise e


def upload_bundle(bundle_name: str, version: str, root_path: str, exist_flag: bool, org_name: str = "MONAI"):
    if exist_flag is False:
        # need to create bundle first
        create_bundle_to_huggingface(bundle_name=bundle_name, org_name=org_name)
    # upload version
    upload_version_to_huggingface(bundle_name=bundle_name, version=version, root_path=root_path, org_name=org_name)
    # access link
    access_link = f"https://huggingface.co/{org_name}/{bundle_name}/tree/{version}"

    return access_link
