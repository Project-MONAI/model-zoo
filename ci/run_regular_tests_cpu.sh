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
#   BUILD_TYPE:   all, tests to execute

set -ex
BUILD_TYPE=all

if [[ $# -eq 1 ]]; then
    BUILD_TYPE=$1

elif [[ $# -gt 1 ]]; then
    echo "ERROR: too many parameters are provided"
    exit 1
fi

init_pipenv() {
    echo "initializing pip environment: $1"
    pipenv install -r $1
    export PYTHONPATH=$PWD
}

remove_pipenv() {
    echo "removing pip environment"
    pipenv --rm
    rm Pipfile Pipfile.lock
}

verify_release_bundle() {
    echo 'Run verify bundle...'
    init_pipenv requirements-dev.txt
    # get all bundles
    download_path="download"
    bundle_list=$(pipenv run python $(pwd)/ci/get_all_latest_bundle.py --model_info $(pwd)/models/model_info.json --p "$download_path")
    pipenv run python $(pwd)/ci/prepare_schema.py --l "$bundle_list" --p "$download_path"
    echo $bundle_list
    for bundle in $bundle_list;
    do
        init_pipenv requirements-dev.txt
        # get required libraries according to the bundle's metadata file
        requirements=$(pipenv run python $(pwd)/ci/get_bundle_requirements.py --b "$bundle" --p "$download_path")
        if [ ! -z "$requirements" ]; then
            echo "install required libraries for bundle: $bundle"
            pipenv install -r "$requirements"
        fi
        # verify bundle
        pipenv run python $(pwd)/ci/verify_bundle.py -b "$bundle" -p "$download_path" -m "regular"  # regular tests on cpu
        remove_pipenv
    done
    fi
}


case $BUILD_TYPE in

    all)
        echo "Run all tests..."
        verify_release_bundle
        ;;
    *)
        echo "ERROR: unknown parameter: $BUILD_TYPE"
        ;;
esac
