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

# Bunles that requires special python version
declare -A bundle_python_versions=(
    ["retinalOCT_RPD_segmentation"]="3.9"
)
DEFAULT_PYTHON_VERSION_FOR_VENV="3.10"

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

    if [[ -z "$CONDA_SHLVL" || "$CONDA_SHLVL" -eq 0 ]]; then
        if [ -n "$CONDA_EXE" ]; then
            source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
        elif [ -n "$MINICONDA_PATH_0" ] && [ -f "$MINICONDA_PATH_0/etc/profile.d/conda.sh" ]; then
            source "$MINICONDA_PATH_0/etc/profile.d/conda.sh"
        else
            echo "Warning: Could not reliably source conda.sh for Conda activation."
        fi
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
                        echo "Bundle '$bundle' requires Python $required_python_version (specified) for GPU. Using Conda."
                        init_conda_env "$required_python_version" "$bundle"
                        active_conda_env_for_bundle="conda_env_${bundle}"
                        conda activate "$active_conda_env_for_bundle"
                    else
                        echo "Bundle '$bundle' using default Python ${DEFAULT_PYTHON_VERSION_FOR_VENV} venv."
                        init_venv
                    fi
                    # Check if the requirements file exists and is not empty
                    if [ -s "$requirements_file" ]; then
                        echo "install required libraries for bundle: $bundle"
                        pip install $include_pre_release -r "$requirements_file"
                    fi
                    # verify bundle
                    python $(pwd)/ci/verify_bundle.py -b "$bundle" -m "min"  # min tests on cpu
                    # cleanup
                    if $use_conda_for_bundle
                    then
                        remove_conda_env "$active_conda_env_for_bundle"
                    else
                        remove_venv
                    fi
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
