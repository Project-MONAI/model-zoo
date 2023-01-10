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

from monai.bundle import get_all_bundles_list


def main(model_info: str):

    bundle_names = ""
    bundle_list = get_all_bundles_list()
    bundle_list = [l[0] for l in bundles_list]

    for bundle in bundle_list:
        bundle_names += f"{bundle} "
    print(bundle_names)


if __name__ == "__main__":
    main()