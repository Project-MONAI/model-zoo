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
import sys

from bundle_custom_data import install_dependency_dict
from utils import get_json_dict

ALLOW_MONAI_RC = os.environ.get("ALLOW_MONAI_RC", "false").lower() in ("true", "1", "t", "y", "yes")


def increment_version(version):
    """
    Split the version into components, assume missing parts are '0'.

    Args:
        version (str): The version string to increment.

    Examples:
        print(increment_version("1.3.2"))  # Expected output: 1.3.3
        print(increment_version("1.4"))    # Expected output: 1.4.1
        print(increment_version("1"))      # Expected output: 1.0.1
    """
    version = str(version)
    parts = version.split(".")
    # Extend the list with zeros to handle cases like "1" or "1.4".
    while len(parts) < 3:
        parts.append("0")

    # Convert all parts to integers.
    parts = list(map(int, parts))

    # Increment the last part.
    parts[-1] += 1

    # Join the parts back into a version string.
    # This trims trailing '.0's, returning the version in its original format
    result_version = ".".join(map(str, parts)).rstrip(".0")

    return result_version


def get_requirements(bundle, models_path, requirements_file):
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
            if not ALLOW_MONAI_RC:
                lib_monai_req = f"monai=={monai_version}"
            else:
                lib_monai_req = f"monai>={monai_version}rc1,<{increment_version(monai_version)}"
                print(f"ALLOW_MONAI_RC is set to true, the version range is {lib_monai_req}", file=sys.stderr)
            libs.append(lib_monai_req)
        if "pytorch_version" in metadata.keys():
            pytorch_version = metadata["pytorch_version"]
            libs.append(f"torch=={pytorch_version}")
        if "numpy_version" in metadata.keys():
            numpy_version = metadata["numpy_version"]
            libs.append(f"numpy=={numpy_version}")
        for package_key in ["optional_packages_version", "required_packages_version"]:
            if package_key in metadata.keys():
                optional_dict = metadata[package_key]
                for name, version in optional_dict.items():
                    libs.append(f"{name}=={version}")

        if len(libs) > 0:
            with open(requirements_file, "w") as f:
                for line in libs:
                    f.write(f"{line}\n")


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
    parser.add_argument("--requirements_file", type=str, help="output filename for requirements.")
    args = parser.parse_args()
    bundle = args.b
    models_path = args.p
    get_script = args.get_script
    if get_script is True:
        get_install_script(bundle)
    else:
        requirements_file = args.requirements_file
        get_requirements(bundle, models_path, requirements_file)
