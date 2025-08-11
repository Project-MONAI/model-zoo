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
#   BUILD_TYPE:   all/specific_test_name, tests to execute

set -ex
BUILD_TYPE=all
export ALLOW_MONAI_RC=true

if [[ $# -eq 1 ]]; then
    BUILD_TYPE=$1

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
    for dir in /opt/hostedtoolcache/*; do
        if [[ $dir != "/opt/hostedtoolcache/Python" ]]; then
            rm -rf "$dir"
        fi
    done
    echo 'Run verify bundle...'
    pip install -r requirements.txt
    # install extra dependencies for get changed bundle
    pip install jsonschema gdown pyyaml
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
            echo $bundle_list
            for bundle in $bundle_list;
            do
                if is_excluded "$bundle"; then
                    echo "skip '$bundle' cpu premerge tests."
                else
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
                    # verify bundle
                    python $(pwd)/ci/verify_bundle.py -b "$bundle" -m "min"  # min tests on cpu
                    remove_venv
                fi
            done
        else
            echo "this pull request does not change any bundles, skip verify."
        fi
    else
        echo "this pull request does not change any files in 'models', skip verify."
    fi
    # check hf models
    hf_model_changes=$(git diff --name-only $head_ref origin/dev -- hf_models)
    if [ ! -z "$hf_model_changes" ]
    then
        # get all changed hf models
        hf_model_list=$(python $(pwd)/ci/get_changed_bundle.py --f "$hf_model_changes" --hf_model True)
        if [ ! -z "$hf_model_list" ]
        then
            python $(pwd)/ci/prepare_schema.py --l "$hf_model_list" --p "hf_models"
            echo $hf_model_list
            for hf_model in $hf_model_list;
            do
                echo "verify hf model: $hf_model"
                # verify hf model
                python $(pwd)/ci/verify_hf_model.py -b "$hf_model"
            done
        else
            echo "this pull request does not change any hf models, skip verify."
        fi
    else
        echo "this pull request does not change any hf models, skip verify."
    fi
}


case $BUILD_TYPE in

    all)
        echo "Run all tests..."
        verify_bundle
        ;;
    changed)
        echo "Run changed tests..."
        verify_bundle
        ;;
    *)
        echo "ERROR: unknown parameter: $BUILD_TYPE"
        ;;
esac
