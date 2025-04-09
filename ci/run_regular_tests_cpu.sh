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
export ALLOW_MONAI_RC=true

if [[ $# -eq 1 ]]; then
    bundle=$1

elif [[ $# -gt 1 ]]; then
    echo "ERROR: too many parameters are provided"
    exit 1
fi

# Usually, CPU test is required, but for some bundles that are too large to run in Github Actions, we can exclude them.
exclude_test_list=("maisi_ct_generative")
is_excluded() {
    for item in "${exclude_test_list[@]}"; do  # Use exclude_test_list here
        if [ "$1" == "$item" ]; then
            return 0 # Return true (0) if excluded
        fi
    done
    return 1 # Return false (1) if not excluded
}

verify_release_bundle() {
    echo 'Run verify bundle...'
    if is_excluded "$bundle"; then
        echo "skip '$bundle' weekly cpu tests."
    else
        download_path="download"
        pip install git+https://github.com/Project-MONAI/MONAI.git@dev  # project-monai/model-zoo issue #505
        pip install jsonschema gdown huggingface_hub==0.29.3
        # download bundle from releases
        python $(pwd)/ci/download_latest_bundle.py --b "$bundle" --models_path $(pwd)/models --p "$download_path"
        # get required libraries according to the bundle's metadata file
        requirements_file="requirements_$bundle.txt"
        python $(pwd)/ci/get_bundle_requirements.py --b "$bundle" --requirements_file "$requirements_file"
        # check if ALLOW_MONAI_RC is set to 1, if so, append --pre to the pip install command
        if [ $ALLOW_MONAI_RC = true ]; then
            include_pre_release="--pre"
        else
            include_pre_release=""
        fi
        # Check if the requirements file exists and is not empty
        if [ -s "$requirements_file" ]; then
            echo "install required libraries for bundle: $bundle"
            pip install $include_pre_release -r "$requirements_file"
        fi
        # verify bundle
        python $(pwd)/ci/verify_bundle.py -b "$bundle" -p "$download_path" -m "regular"  # regular tests on cpu
    fi
}


case $bundle in

    *)
        echo "Check bundle: $bundle"
        verify_release_bundle
        ;;
esac
