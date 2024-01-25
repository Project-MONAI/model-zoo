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

from monai.bundle import download
from utils import get_checksum, get_hash_func, get_latest_version, get_version_checksum


def download_latest_bundle(bundle_name: str, models_path: str, download_path: str):
    model_info_path = os.path.join(models_path, "model_info.json")
    version = get_latest_version(bundle_name=bundle_name, model_info_path=model_info_path)
    checksum = get_version_checksum(bundle_name=bundle_name, version=version, model_info_path=model_info_path)
    download(name=bundle_name, source="monaihosting", version=version, bundle_dir=download_path)

    # verify checksum
    hash_func = get_hash_func(hash_type="sha1")
    bundle_zip_name = f"{bundle_name}_v{version}.zip"
    downloaded_checksum = get_checksum(dst_path=os.path.join(download_path, bundle_zip_name), hash_func=hash_func)
    if checksum != downloaded_checksum:
        raise ValueError(f"Checksum of {bundle_zip_name} is not correct.")
    print(f"Checksum of downloaded bundle {bundle_name} is correct.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-b", "--b", type=str, help="bundle name.")
    parser.add_argument("-models_path", "--models_path", type=str, help="models path.")
    parser.add_argument("-p", "--p", type=str, help="download path.")
    args = parser.parse_args()
    bundle_name = args.b
    models_path = args.models_path
    download_path = args.p
    download_latest_bundle(bundle_name, models_path, download_path)
