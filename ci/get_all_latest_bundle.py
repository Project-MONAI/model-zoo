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

from utils import get_latest_bundle_list


def main(model_info, download_path):

    bundle_names = ""
    bundle_list = get_latest_bundle_list(model_info=model_info, download_path=download_path)

    for bundle in bundle_list:
        bundle_names += f"{bundle} "
    print(bundle_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-model_info", "--model_info", type=str, help="model_info.json path")
    parser.add_argument("-p", "--p", type=str, help="download path")
    args = parser.parse_args()
    download_path = args.p
    model_info = args.model_info
    main(model_info, download_path)
