#! /bin/bash

set -v

eval "$(conda shell.bash hook)"
conda activate monai

homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

BUNDLE="$(cd "$homedir/.." && pwd)"

echo "Bundle root: $BUNDLE"

export PYTHONPATH="$BUNDLE"

# set this to something else to use different numbered GPUs on your system
export CUDA_VISIBLE_DEVICES="0,1"

# seems to resolve some multiprocessing issues with certain libraries
export OMP_NUM_THREADS=1

CKPT=none

# need to change this if you have multiple nodes or not 2 GPUs
PYTHON="torchrun --standalone --nnodes=1 --nproc_per_node=2"

CONFIG="['$BUNDLE/configs/train.yaml','$BUNDLE/configs/multi_gpu_train.yaml']"

$PYTHON -m monai.bundle run \
    --meta_file $BUNDLE/configs/metadata.json \
    --logging_file $BUNDLE/configs/logging.conf \
    --config_file "$CONFIG" \
    --bundle_root $BUNDLE \
    $@
