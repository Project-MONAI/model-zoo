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

declare -A bundle_python_versions=(
    ["retinalOCT_RPD_segmentation"]="3.9"
)
DEFAULT_PYTHON_VERSION_FOR_VENV="3.10"

install_common_deps_in_activated_env() {
    python -m pip install --upgrade pip wheel
    python -m pip install --upgrade setuptools
    python -m pip install jsonschema gdown pyyaml parameterized fire
    export PYTHONPATH=$PWD
}

init_venv() {
    if [ ! -d "model_zoo_venv" ]; then  # Check if the venv directory does not exist
        echo "initializing pip environment"
        python -m venv model_zoo_venv
        source model_zoo_venv/bin/activate
        install_common_deps_in_activated_env
    else
        echo "Virtual environment model_zoo_venv already exists. Activating..."
        source model_zoo_venv/bin/activate
        install_common_deps_in_activated_env
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

init_conda_env() {
    local python_version_to_create="$1"
    local bundle_identifier="$2"
    local conda_env_name="conda_env_${bundle_identifier}"

    # Always source conda.sh to ensure conda activate is available
    if [ -n "$CONDA_EXE" ] && [ -f "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh" ]; then
        source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
    elif [ -n "$MINICONDA_PATH_0" ] && [ -f "$MINICONDA_PATH_0/etc/profile.d/conda.sh" ]; then
        source "$MINICONDA_PATH_0/etc/profile.d/conda.sh"
    else
        echo "Warning: Could not reliably source conda.sh for Conda activation."
    fi

    if conda env list | grep -q "^${conda_env_name}[[:space:]]"; then
        echo "Conda env '$conda_env_name' already exists. Removing for a clean start..."
        conda env remove -n "$conda_env_name" -y
    fi

    conda create -n "$conda_env_name" python="$python_version_to_create" -y
    conda activate "$conda_env_name"
    install_common_deps_in_activated_env
    conda deactivate 2>/dev/null || true
}

remove_conda_env() {
    local conda_env_name_to_remove="$1"
    if [ -z "$conda_env_name_to_remove" ]; then
        echo "Warning: No Conda env name provided to remove_conda_env."
        return
    fi
    echo "Deactivating and removing Conda environment: $conda_env_name_to_remove"
    conda deactivate 2>/dev/null || true
    conda env remove -n "$conda_env_name_to_remove" -y
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
            # determine if conda env should be used for the bundle
            active_conda_env_for_bundle=""
            required_python_version="${bundle_python_versions[$bundle]}"
            use_conda_for_bundle=false
            if [[ -n "$required_python_version" && "$required_python_version" != "$DEFAULT_PYTHON_VERSION_FOR_VENV" ]]
            then
                use_conda_for_bundle=true
            fi
            if $use_conda_for_bundle
            then
                init_conda_env "$required_python_version" "$bundle"
                active_conda_env_for_bundle="conda_env_${bundle}"
                conda activate "$active_conda_env_for_bundle"
            else
                init_venv
            fi
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
            # cleanup
            if $use_conda_for_bundle
            then
                remove_conda_env "$active_conda_env_for_bundle"
            else
                remove_venv
            fi
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
