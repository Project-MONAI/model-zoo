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
#   $1 - Dist flag (True/False)

dist_flag=$1

set -ex

export ALLOW_MONAI_RC=true

if [[ $# -gt 1 ]]; then
    echo "ERROR: too many parameters are provided"
    exit 1
fi

init_venv() {
    if [ ! -d "model_zoo_venv" ]; then  # Check if the venv directory does not exist
        echo "initializing pip environment"
        python -m venv model_zoo_venv
        source model_zoo_venv/bin/activate
        pip install --upgrade pip wheel
        pip install --upgrade setuptools
        pip install jsonschema gdown pyyaml parameterized fire
        export PYTHONPATH=$PWD
    else
        echo "Virtual environment model_zoo_venv already exists. Activating..."
        source model_zoo_venv/bin/activate
        pip install --upgrade pip wheel
        pip install --upgrade setuptools
        pip install jsonschema gdown pyyaml parameterized fire
        export PYTHONPATH=$PWD
    fi
}

remove_venv() {
    if [ -d "model_zoo_venv" ]; then  # Check if the venv directory exists
        echo "Removing virtual environment..."
        deactivate 2>/dev/null || true  # Deactivate venv, ignore errors if not activated
        rm -rf model_zoo_venv  # Remove the venv directory
    else
        echo "Virtual environment not found. Skipping removal."
    fi
}

verify_bundle() {
    echo 'Run verify bundle...'
    head_ref=$(git rev-parse HEAD)
    git fetch origin dev $head_ref
    # achieve all changed files in 'models'
    changes=$(git diff --name-only $head_ref origin/dev -- models)
    if [ ! -z "$changes" ]
    then
        # get all changed bundles
        bundle_list=$(python $(pwd)/ci/get_changed_bundle.py --f "$changes")
        if [ ! -z "$bundle_list" ]
        then
            python $(pwd)/ci/prepare_schema.py --l "$bundle_list"
        for bundle in $bundle_list;
        do
            # get required libraries according to the bundle's metadata file
            requirements_file="requirements_$bundle.txt"
            python $(pwd)/ci/get_bundle_requirements.py --b "$bundle" --requirements_file "$requirements_file"
            # check if ALLOW_MONAI_RC is set to 1, if so, append --pre to the pip install command
            if [ $ALLOW_MONAI_RC = true ]; then
                include_pre_release="--pre"
            else
                include_pre_release=""
            fi
            init_venv
            # Check if the requirements file exists and is not empty
            if [ -s "$requirements_file" ]; then
                echo "install required libraries for bundle: $bundle"
                pip install $include_pre_release -r "$requirements_file"
            fi
            # get extra install script if exists
            extra_script=$(python $(pwd)/ci/get_bundle_requirements.py --b "$bundle" --get_script True)
            if [ ! -z "$extra_script" ]; then
                echo "install extra libraries with script: $extra_script"
                bash $extra_script
            fi
            # verify bundle
            python $(pwd)/ci/verify_bundle.py --b "$bundle"
            test_cmd="python $(pwd)/ci/unit_tests/runner.py --b \"$bundle\""
            if [ "$dist_flag" = "True" ]; then
                test_cmd="torchrun $(pwd)/ci/unit_tests/runner.py --b \"$bundle\" --dist True"
            fi
            eval $test_cmd
            remove_venv
        done
        else
            echo "this pull request does not change any bundles, skip verify."
        fi
    else
        echo "this pull request does not change any files in 'models', skip verify."
        remove_venv
    fi
}

verify_bundle
