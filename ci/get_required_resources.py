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


def get_changed_bundle_list(changed_dirs, root_path: str = "models"):
    """
    This function is used to return all bundle names that have changed files.
    If a bundle is totally removed, it will be ignored (since it not exists).
    A similar function is implemented in "utils.py", but that file requires monai.
    In order to minimize the required libraries when running this file, this function
    is defined.

    """
    bundles = [f.name for f in os.scandir(root_path) if f.is_dir()]

    changed_bundle_list = []
    for sub_dir in changed_dirs:
        for bundle in bundles:
            bundle_path = os.path.join(root_path, bundle)
            if os.path.commonpath([bundle_path]) == os.path.commonpath([bundle_path, sub_dir]):
                changed_bundle_list.append(bundle)

    return list(set(changed_bundle_list))


def get_required_resources(changed_dirs):
    """
    This function is used to determine if GPU or multi-GPU resources are needed.
    A string which contains two flags will be printed, and can be used in shell scripts.
    """
    gpu_flag, mgpu_flag = False, False
    root_path, unit_test_path = "models", "ci/unit_tests"
    bundle_list = get_changed_bundle_list(changed_dirs, root_path=root_path)
    if len(bundle_list) > 0:
        gpu_flag = True
        for bundle in bundle_list:
            mgpu_test_file = os.path.join(unit_test_path, f"test_{bundle}_dist.py")
            if os.path.exists(mgpu_test_file):
                mgpu_flag = True
    print(f"{gpu_flag} {mgpu_flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--f", type=str, help="changed files.")
    args = parser.parse_args()
    changed_dirs = args.f.splitlines()
    get_required_resources(changed_dirs)
