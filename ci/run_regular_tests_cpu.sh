#!/bin/bash
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Argument(s):
#   bundle:   bundle to be tested

set -ex
bundle=""

if [[ $# -eq 1 ]]; then
    bundle=$1

elif [[ $# -gt 1 ]]; then
    echo "ERROR: too many parameters are provided"
    exit 1
fi

verify_release_bundle() {
    echo 'Run verify bundle...'
    # get all bundles
    download_path="download"
    pip install -r requirements-dev.txt
    # download bundle from releases
    python $(pwd)/ci/download_latest_bundle.py --b "$bundle" --models_path $(pwd)/models --p "$download_path"
    # get required libraries according to the bundle's metadata file
    requirements=$(python $(pwd)/ci/get_bundle_requirements.py --b "$bundle" --p "$download_path")
    if [ ! -z "$requirements" ]; then
        echo "install required libraries for bundle: $bundle"
        pip install -r "$requirements"
    fi
    # verify bundle
    python $(pwd)/ci/verify_bundle.py -b "$bundle" -p "$download_path" -m "regular"  # regular tests on cpu
}


case $bundle in

    *)
        echo "Check bundle: $bundle"
        verify_release_bundle
        ;;
esac
