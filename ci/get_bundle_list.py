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

from utils import get_sub_folders

# new added bundles should temporarily be added to this list, and remove until they can be downloaded successfully
EXCLUDE_LIST = []


def main(models_path):
    bundle_list = get_sub_folders(root_dir=models_path)
    bundle_list = [b for b in bundle_list if b not in EXCLUDE_LIST]
    print(bundle_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-models_path", "--models_path", type=str, help="models path.")
    args = parser.parse_args()
    models_path = args.models_path
    main(models_path)
