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
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

from monai.utils import look_up_option

SUPPORTED_HASH_TYPES = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256, "sha512": hashlib.sha512}

def get_sub_folders(root_dir: str):
    """
    This function is used to get all sub folders (as a list) within the root_dir.
    """
    sub_folder_list = [f.name for f in os.scandir(root_dir) if f.is_dir()]

    return sub_folder_list

def get_changed_bundle_list(changed_dirs: List[str], root_path: str = "models"):
    """
    This function is used to return all bundle names that have changed files.

    """
    bundles = get_sub_folders(root_path)

    changed_bundle_list = []
    for sub_dir in changed_dirs:
        for bundle in bundles:
            bundle_path = os.path.join(root_path, bundle)
            if os.path.commonpath([bundle_path]) == os.path.commonpath([bundle_path, sub_dir]):
                changed_bundle_list.append(bundle)
