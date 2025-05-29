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
declare -A bundle_python_versions=(
    ["retinalOCT_RPD_segmentation"]="3.9"
)
DEFAULT_PYTHON_VERSION_FOR_VENV="3.10"

if [[ $# -gt 1 ]]; then
    echo "ERROR: too many parameters are provided" >&2
    exit 1
fi

echo "CI GPU Job starting... Dist flag: $dist_flag"
echo "Global Python (expected ${DEFAULT_PYTHON_VERSION_FOR_VENV} from YAML): $(python -V)"
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda command not found. Ensure Miniconda was installed by the YAML." >&2
    exit 1
else
    echo "Conda executable found: $(command -v conda)" >&2
fi

# Common dependencies to install in any activated environment
install_common_deps_in_activated_env() {
    echo "Installing common dependencies in activated environment..." >&2
    python -m pip install --upgrade pip wheel >&2
    python -m pip install --upgrade setuptools >&2
    python -m pip install jsonschema gdown pyyaml parameterized fire >&2
    export PYTHONPATH=$PWD
}


init_venv() {
    if [ ! -d "model_zoo_venv" ]; then
        echo "Initializing pip environment (model_zoo_venv for Python ${DEFAULT_PYTHON_VERSION_FOR_VENV})" >&2
        python -m venv model_zoo_venv
        source model_zoo_venv/bin/activate
        install_common_deps_in_activated_env
    else
        echo "Virtual environment model_zoo_venv already exists. Activating..." >&2
        source model_zoo_venv/bin/activate
        install_common_deps_in_activated_env
    fi
}

remove_venv() {
    if [ -d "model_zoo_venv" ]; then
        echo "Removing virtual environment model_zoo_venv..." >&2
        deactivate 2>/dev/null || true
        rm -rf model_zoo_venv
    else
        echo "Virtual environment model_zoo_venv not found. Skipping removal." >&2
    fi
}

# Conda environment functions
init_conda_env() {
    local python_version_to_create="$1"
    local bundle_identifier="$2"
    local conda_env_name="conda_env_${bundle_identifier}"

    echo "Initializing Conda environment with Python $python_version_to_create for GPU bundle '$bundle_identifier'..." >&2

    if [[ -z "$CONDA_SHLVL" || "$CONDA_SHLVL" -eq 0 ]]; then
        if [ -n "$CONDA_EXE" ]; then
            source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
        elif [ -n "$MINICONDA_PATH_0" ] && [ -f "$MINICONDA_PATH_0/etc/profile.d/conda.sh" ]; then
            source "$MINICONDA_PATH_0/etc/profile.d/conda.sh"
        else
            echo "Warning: Could not reliably source conda.sh for Conda activation." >&2
        fi
    fi

    if conda env list | grep -q "^${conda_env_name}[[:space:]]"; then
        echo "Conda env '$conda_env_name' already exists. Removing for a clean start..." >&2
        conda env remove -n "$conda_env_name" -y >&2
    fi

    conda create -n "$conda_env_name" python="$python_version_to_create" -y >&2
    conda activate "$conda_env_name"
    install_common_deps_in_activated_env
    conda deactivate 2>/dev/null || true

    echo "$conda_env_name"
}

remove_conda_env() {
    local conda_env_name_to_remove="$1"
    if [ -z "$conda_env_name_to_remove" ]; then
        echo "Warning: No Conda env name provided to remove_conda_env." >&2
        return
    fi
    echo "Deactivating and removing Conda environment: $conda_env_name_to_remove" >&2
    conda deactivate 2>/dev/null || true
    conda env remove -n "$conda_env_name_to_remove" -y >&2
}

verify_bundle() {
    echo 'Run verify bundle...' >&2

    # Source conda.sh to initialize Conda for the main script shell
    if [[ -z "$CONDA_SHLVL" || "$CONDA_SHLVL" -eq 0 ]]; then
        if [ -n "$CONDA_EXE" ]; then
            echo "Sourcing conda.sh from CONDA_EXE path ($(dirname "$CONDA_EXE")) for main shell..." >&2
            source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
        else
            echo "Error: CONDA_EXE is not set. Cannot reliably source conda.sh for main shell. Exiting." >&2
            exit 1
        fi
    else
        echo "Conda shell already initialized at level: $CONDA_SHLVL" >&2
    fi

    head_ref=$(git rev-parse HEAD)
    git fetch origin dev $head_ref

    changes=$(git diff --name-only $head_ref origin/dev -- models)

    if [ ! -z "$changes" ]
    then
        echo "Detected changes in 'models': $changes" >&2
        bundle_list=$(python "$(pwd)/ci/get_changed_bundle.py" --f "$changes")
        if [ ! -z "$bundle_list" ]
        then
            python "$(pwd)/ci/prepare_schema.py" --l "$bundle_list"
            echo "GPU Bundles to process: $bundle_list" >&2
            for bundle in $bundle_list
            do
                echo "Processing GPU bundle: $bundle" >&2
                requirements_file="requirements_$bundle.txt"
                python "$(pwd)/ci/get_bundle_requirements.py" --b "$bundle" --requirements_file "$requirements_file"

                # check if ALLOW_MONAI_RC is set to 1, if so, append --pre to the pip install command
                if [ "$ALLOW_MONAI_RC" = true ]
                then
                    include_pre_release="--pre"
                else
                    include_pre_release=""
                fi

                # determine if conda should be used for the bundle
                required_python_version="${bundle_python_versions[$bundle]}"
                use_conda_for_bundle=false
                if [[ -n "$required_python_version" && "$required_python_version" != "$DEFAULT_PYTHON_VERSION_FOR_VENV" ]]
                then
                    use_conda_for_bundle=true
                fi

                if $use_conda_for_bundle
                then
                    echo "Bundle '$bundle' requires Python $required_python_version (specified) for GPU. Using Conda." >&2
                    init_conda_env "$required_python_version" "$bundle"
                    active_conda_env_for_bundle="conda_env_${bundle}"
                    conda activate "$active_conda_env_for_bundle"
                else
                    echo "Bundle '$bundle' using default Python ${DEFAULT_PYTHON_VERSION_FOR_VENV} venv for GPU." >&2
                    init_venv
                fi
                if [ -s "$requirements_file" ]
                then
                    echo "Installing requirements from $requirements_file for $bundle" >&2
                    python -m pip install $include_pre_release -r "$requirements_file" >&2
                fi

                extra_script=$(python "$(pwd)/ci/get_bundle_requirements.py" --b "$bundle" --get_script True)
                if [ ! -z "$extra_script" ]
                then
                    echo "Installing extra libraries for GPU with script: $extra_script" >&2
                    bash "$extra_script"
                fi

                echo "Verifying GPU bundle: $bundle" >&2
                python "$(pwd)/ci/verify_bundle.py" --b "$bundle"

                test_cmd="python $(pwd)/ci/unit_tests/runner.py --b \"$bundle\""
                if [ "$dist_flag" = "True" ]; then
                    test_cmd="torchrun $(pwd)/ci/unit_tests/runner.py --b \"$bundle\" --dist True"
                fi
                echo "Executing GPU test command: $test_cmd" >&2
                eval $test_cmd

                if $use_conda_for_bundle
                then
                    remove_conda_env "$active_conda_env_for_bundle"
                else
                    remove_venv
                fi
                echo "Finished processing GPU bundle: $bundle" >&2
            done
        else
            echo "No bundles found by get_changed_bundle.py for GPU processing." >&2
        fi
    else
        echo "No changes in 'models' directory. Skipping GPU bundle verification." >&2
        remove_venv
    fi
}

verify_bundle
