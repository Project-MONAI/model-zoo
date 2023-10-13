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

from bundle_custom_data import install_dependency_dict
from utils import get_json_dict


def get_requirements(bundle, models_path):
    """
    This function is used to produce a requirements txt file, and print a string
    which shows the filename. The printed string can be used in shell scripts.
    """
    bundle_path = os.path.join(models_path, bundle)
    meta_file_path = os.path.join(bundle_path, "configs/metadata.json")
    if os.path.exists(meta_file_path):
        metadata = get_json_dict(meta_file_path)
        libs = []
        if "monai_version" in metadata.keys():
            monai_version = metadata["monai_version"]
            libs.append(f"monai=={monai_version}")
        if "pytorch_version" in metadata.keys():
            pytorch_version = metadata["pytorch_version"]
            libs.append(f"torch=={pytorch_version}")
        if "numpy_version" in metadata.keys():
            numpy_version = metadata["numpy_version"]
            libs.append(f"numpy=={numpy_version}")
        if "optional_packages_version" in metadata.keys():
            optional_dict = metadata["optional_packages_version"]
            for name, version in optional_dict.items():
                libs.append(f"{name}=={version}")

        if len(libs) > 0:
            requirements_file_name = f"requirements_{bundle}.txt"
            with open(requirements_file_name, "w") as f:
                for line in libs:
                    f.write(f"{line}\n")
            print(requirements_file_name)


def get_install_script(bundle):
    # install extra dependencies if needed
    script_path = ""
    if bundle in install_dependency_dict.keys():
        script_path = install_dependency_dict[bundle]
    print(script_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-b", "--b", type=str, help="bundle name.")
    parser.add_argument("-p", "--p", type=str, default="models", help="models path.")
    parser.add_argument("--get_script", type=bool, default=False, help="whether to get the install script.")
    args = parser.parse_args()
    bundle = args.b
    models_path = args.p
    get_script = args.get_script
    if get_script is True:
        get_install_script(bundle)
    else:
        get_requirements(bundle, models_path)
