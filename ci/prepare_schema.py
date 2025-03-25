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

from utils import prepare_schema


def main(bundle_list, models_path):
    if "hf_models" in models_path:
        hf_model = True
    else:
        hf_model = False
    prepare_schema(bundle_list, root_path=models_path, hf_model=hf_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-l", "--l", type=str, help="bundle list.")
    parser.add_argument("-p", "--p", type=str, default="models", help="models path.")
    args = parser.parse_args()
    bundle_list = args.l.split(" ")
    bundle_list = [f for f in bundle_list if f != ""]
    models_path = args.p
    main(bundle_list, models_path)
