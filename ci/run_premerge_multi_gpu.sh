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
ALLOW_MONAI_RC=true

if [[ $# -eq 1 ]]; then
    BUILD_TYPE=$1

elif [[ $# -gt 1 ]]; then
    echo "ERROR: too many parameters are provided"
    exit 1
fi

init_pipenv() {
    echo "initializing pip environment: $1"
    pipenv install update pip wheel
    pipenv install --python=3.8 -r $1
    export PYTHONPATH=$PWD
}

remove_pipenv() {
    echo "removing pip environment"
    pipenv --rm
    rm Pipfile Pipfile.lock
}

verify_bundle() {
    echo 'Run verify bundle...'
    init_pipenv requirements-dev.txt
    head_ref=$(git rev-parse HEAD)
    git fetch origin dev $head_ref
    # achieve all changed files in 'models'
    changes=$(git diff --name-only $head_ref origin/dev -- models)
    if [ ! -z "$changes" ]
    then
        # get all changed bundles
        bundle_list=$(pipenv run python $(pwd)/ci/get_changed_bundle.py --f "$changes")
        if [ ! -z "$bundle_list" ]
        then
            pipenv run python $(pwd)/ci/prepare_schema.py --l "$bundle_list"
            for bundle in $bundle_list;
            do
                init_pipenv requirements-dev.txt
                # get required libraries according to the bundle's metadata file
                requirements=$(pipenv run python $(pwd)/ci/get_bundle_requirements.py --b "$bundle")
                # check if ALLOW_MONAI_RC is set to 1, if so, append --pre to the pip install command
                if [ $ALLOW_MONAI_RC = true ]; then
                    include_pre_release="--pre"
                else
                    include_pre_release=""
                fi
                if [ ! -z "$requirements" ]; then
                    echo "install required libraries for bundle: $bundle"
                    pipenv install "$include_pre_release" -r "$requirements"
                fi
                # get extra install script if exists
                extra_script=$(pipenv run python $(pwd)/ci/get_bundle_requirements.py --b "$bundle" --get_script True)
                if [ ! -z "$extra_script" ]; then
                    echo "install extra libraries with script: $extra_script"
                    bash $extra_script
                fi
                # do multi gpu based unit tests
                pipenv run torchrun $(pwd)/ci/unit_tests/runner.py --b "$bundle" --dist True
                remove_pipenv
            done
        else
            echo "this pull request does not change any bundles, skip verify."
        fi
    else
        echo "this pull request does not change any files in 'models', skip verify."
        remove_pipenv
    fi
}

case $BUILD_TYPE in

    all)
        echo "Run all tests..."
        verify_bundle
        ;;

    verify_bundle)
        verify_bundle
        ;;

    *)
        echo "ERROR: unknown parameter: $BUILD_TYPE"
        ;;
esac
