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

if [[ $# -eq 1 ]]; then
    BUILD_TYPE=$1

elif [[ $# -gt 1 ]]; then
    echo "ERROR: too many parameters are provided"
    exit 1
fi

DOCKER_CONTAINER="projectmonai/monai:latest"

verify_tensorrt() {
    echo 'Run verify tensorrt bundle...'
    head_ref=$(git rev-parse HEAD)
    git fetch origin dev $head_ref
    # achieve all changed files in 'models'
    changes=$(git diff --name-only $head_ref origin/dev -- models)
    if [ ! -z "$changes" ]
    then
        docker run --gpus all -it --ipc=host --rm --net host \
        -v $(pwd):/workspace projectmonai/monai:latest bash -c "cd /workspace; python ci/verify_tensorrt.py --f '$changes'"

    else
        echo "this pull request does not change any files in 'models', skip verify."
    fi
}

case $BUILD_TYPE in

    all)
        echo "Run all tests..."
        verify_tensorrt
        ;;

    verify_tensorrt)
        verify_bundle
        ;;

    *)
        echo "ERROR: unknown parameter: $BUILD_TYPE"
        ;;
esac
