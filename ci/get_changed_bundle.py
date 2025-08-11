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

from utils import get_changed_bundle_list


def get_changed_bundle(changed_dirs):
    """
    This function is used to get all changed bundles, a string which
    contains all bundle names will be printed, and can be used in shell scripts.
    """
    bundle_names = ""
    root_path = "models"
    bundle_list = get_changed_bundle_list(changed_dirs, root_path=root_path)

    for bundle in bundle_list:
        bundle_names += f"{bundle} "
    print(bundle_names)


def get_changed_hf_model(changed_dirs):
    """
    This function is used to get all changed hf models, a string which
    contains all hf model names will be printed, and can be used in shell scripts.
    """
    hf_model_names = ""
    root_path = "hf_models"
    hf_model_list = get_changed_bundle_list(changed_dirs, root_path=root_path)
    for hf_model in hf_model_list:
        hf_model_names += f"{hf_model} "
    print(hf_model_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--f", type=str, help="changed files.")
    parser.add_argument("--hf_model", type=bool, default=False, help="if true, get changed hf models.")
    args = parser.parse_args()
    changed_dirs = args.f.splitlines()
    if args.hf_model:
        get_changed_hf_model(changed_dirs)
    else:
        get_changed_bundle(changed_dirs)
