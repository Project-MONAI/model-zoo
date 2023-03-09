#! /bin/bash

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# script for running all tests
set -e

# output formatting
separator=""
blue=""
green=""
red=""
noColor=""

if [[ -t 1 ]] # stdout is a terminal
then
    separator=$'--------------------------------------------------------------------------------\n'
    blue="$(tput bold; tput setaf 4)"
    green="$(tput bold; tput setaf 2)"
    red="$(tput bold; tput setaf 1)"
    noColor="$(tput sgr0)"
fi

# configuration values
doDryRun=false
doBlackFormat=false
doBlackFix=false
doIsortFormat=false
doIsortFix=false
doFlake8Format=false
doPytypeFormat=false
doCleanup=false
doPrecommit=false

NUM_PARALLEL=1

PY_EXE=${MONAI_MODEL_ZOO_PY_EXE:-$(which python)}

function print_usage {
    echo "runtests.sh [--codeformat] [--autofix] [--black] [--isort] [--flake8] [--pytype]"
    echo "            [--dryrun] [-j number] [--clean] [--precommit] [--help] [--version]"
    echo ""
    echo "MONAI Model Zoo testing utilities."
    echo ""
    echo "Examples:"
    echo "./runtests.sh -f                      # run coding style and static type checking."
    echo "./runtests.sh --autofix               # run automatic code formatting using \"isort\" and \"black\"."
    echo "./runtests.sh --clean                 # clean up temporary files."
    echo ""
    echo "Code style check options:"
    echo "    --black           : perform \"black\" code format checks"
    echo "    --autofix         : format code using \"isort\" and \"black\""
    echo "    --isort           : perform \"isort\" import sort checks"
    echo "    --flake8          : perform \"flake8\" code format checks"
    echo "    --precommit       : perform source code format check and fix using \"pre-commit\""
    echo ""
    echo "Python type check options:"
    echo "    --pytype          : perform \"pytype\" static type checks"
    echo "    -j, --jobs        : number of parallel jobs to run \"pytype\" (default $NUM_PARALLEL)"
    echo ""
    echo "Misc. options:"
    echo "    --dryrun          : display the commands to the screen without running"
    echo "    -f, --codeformat  : shorthand to run all code style and static analysis tests"
    echo "    -c, --clean       : clean temporary files from tests and exit"
    echo "    -h, --help        : show this help message and exit"
    echo "    -v, --version     : show MONAI and system version information and exit"
    echo ""
    echo "${separator}For bug reports and feature requests, please file an issue at:"
    echo "    https://github.com/Project-MONAI/model-zoo/issues/new/choose"
    echo ""
    echo "To choose an alternative python executable, set the environmental variable, \"MONAI_MODEL_ZOO_PY_EXE\"."
    exit 1
}

function check_import {
    echo "Python: ${PY_EXE}"
    ${cmdPrefix}${PY_EXE} -W error -W ignore::DeprecationWarning -c "import monai"
}

function print_monai_version {
    ${cmdPrefix}${PY_EXE} -c 'import monai; monai.config.print_config()'
}

function install_monai {
    echo "Pip installing MONAI basic dependencies"
    ${cmdPrefix}${PY_EXE} -m pip install -r requirements.txt
}

function install_deps {
    echo "Pip installing MONAI development dependencies"
    ${cmdPrefix}${PY_EXE} -m pip install -r requirements-dev.txt
}

function clean_py {
    # remove temporary files (in the directory of this script)
    TO_CLEAN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    echo "Removing temporary files in ${TO_CLEAN}"

    find ${TO_CLEAN} -depth -maxdepth 1 -type d -name ".pytype" -exec rm -r "{}" +
    find ${TO_CLEAN} -depth -maxdepth 1 -type d -name "__pycache__" -exec rm -r "{}" +
}

function print_error_msg() {
    echo "${red}Error: $1.${noColor}"
    echo ""
}

function print_style_fail_msg() {
    echo "${red}Check failed!${noColor}"
    echo "Please run auto style fixes: ${green}./runtests.sh --autofix${noColor}"
}

function is_pip_installed() {
    return $(${PY_EXE} -c "import sys, pkgutil; sys.exit(0 if pkgutil.find_loader(sys.argv[1]) else 1)" $1)
}

if [ -z "$1" ]
then
    print_error_msg "Too few arguments to $0"
    print_usage
fi

# parse arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --dryrun)
            doDryRun=true
        ;;
        -f|--codeformat)
            doBlackFormat=true
            doIsortFormat=true
            doFlake8Format=true
            doPytypeFormat=true
        ;;
        --black)
            doBlackFormat=true
        ;;
        --autofix)
            doIsortFix=true
            doBlackFix=true
            doIsortFormat=true
            doBlackFormat=true
        ;;
        --isort)
            doIsortFormat=true
        ;;
        --flake8)
            doFlake8Format=true
        ;;
        --precommit)
            doPrecommit=true
        ;;
        --pytype)
            doPytypeFormat=true
        ;;
        -j|--jobs)
            NUM_PARALLEL=$2
            shift
        ;;
        -c|--clean)
            doCleanup=true
        ;;
        -h|--help)
            print_usage
        ;;
        *)
            print_error_msg "Incorrect commandline provided, invalid key: $key"
            print_usage
        ;;
    esac
    shift
done

# home directory
homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$homedir"

# python path
export PYTHONPATH="$homedir:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# by default do nothing
cmdPrefix=""

if [ $doDryRun = true ]
then
    echo "${separator}${blue}dryrun${noColor}"

    # commands are echoed instead of ran
    cmdPrefix="dryrun "
    function dryrun { echo "    " "$@"; }
else
    if ! is_pip_installed monai
    then
        install_monai
    fi
    check_import
fi

if [ $doCleanup = true ]
then
    echo "${separator}${blue}clean${noColor}"

    clean_py

    echo "${green}done!${noColor}"
    exit
fi

print_monai_version

if [ $doIsortFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    if [ $doIsortFix = true ]
    then
        echo "${separator}${blue}isort-fix${noColor}"
    else
        echo "${separator}${blue}isort${noColor}"
    fi

    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed isort
    then
        install_deps
    fi
    ${cmdPrefix}${PY_EXE} -m isort --version

    if [ $doIsortFix = true ]
    then
        ${cmdPrefix}${PY_EXE} -m isort "$(pwd)"
    else
        ${cmdPrefix}${PY_EXE} -m isort --check "$(pwd)"
    fi

    isort_status=$?
    if [ ${isort_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${isort_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi


if [ $doBlackFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    if [ $doBlackFix = true ]
    then
        echo "${separator}${blue}black-fix${noColor}"
    else
        echo "${separator}${blue}black${noColor}"
    fi

    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed black
    then
        install_deps
    fi
    ${cmdPrefix}${PY_EXE} -m black --version

    if [ $doBlackFix = true ]
    then
        ${cmdPrefix}${PY_EXE} -m black --skip-magic-trailing-comma "$(pwd)"
    else
        ${cmdPrefix}${PY_EXE} -m black --skip-magic-trailing-comma --check "$(pwd)"
    fi

    black_status=$?
    if [ ${black_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${black_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi


if [ $doFlake8Format = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    echo "${separator}${blue}flake8${noColor}"

    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed flake8
    then
        install_deps
    fi
    ${cmdPrefix}${PY_EXE} -m flake8 --version

    ${cmdPrefix}${PY_EXE} -m flake8 "$(pwd)" --count --statistics

    flake8_status=$?
    if [ ${flake8_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${flake8_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi


if [ $doPrecommit = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    echo "${separator}${blue}pre-commit${noColor}"

    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed pre_commit
    then
        install_deps
    fi
    ${cmdPrefix}${PY_EXE} -m pre_commit run --all-files

    pre_commit_status=$?
    if [ ${pre_commit_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${pre_commit_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi


if [ $doPytypeFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    echo "${separator}${blue}pytype${noColor}"
    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed pytype
    then
        install_deps
    fi
    pytype_ver=$(${cmdPrefix}${PY_EXE} -m pytype --version)
    if [[ "$OSTYPE" == "darwin"* && "$pytype_ver" == "2021."* ]]; then
        echo "${red}pytype not working on macOS 2021 (https://github.com/Project-MONAI/MONAI/issues/2391). Please upgrade to 2022*.${noColor}"
        exit 1
    else
        ${cmdPrefix}${PY_EXE} -m pytype --version

        ${cmdPrefix}${PY_EXE} -m pytype -j ${NUM_PARALLEL} --python-version="$(${PY_EXE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")" "$(pwd)"

        pytype_status=$?
        if [ ${pytype_status} -ne 0 ]
        then
            echo "${red}failed!${noColor}"
            exit ${pytype_status}
        else
            echo "${green}passed!${noColor}"
        fi
    fi
    set -e # enable exit on failure
fi
