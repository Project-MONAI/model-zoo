#! /bin/bash

set -v

eval "$(conda shell.bash hook)"
conda activate monai

homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

BUNDLE="$(cd "$homedir/.." && pwd)"

echo "Bundle root: $BUNDLE"

export PYTHONPATH="$BUNDLE"

export CUDA_VISIBLE_DEVICES="0,1,2,3"

export OMP_NUM_THREADS=1

CKPT=none

PYTHON="torchrun --standalone --nnodes=1 --nproc_per_node=4"

CONFIG="['$BUNDLE/configs/train.yaml','$BUNDLE/configs/multi_gpu_train.yaml']"


$PYTHON -m monai.bundle run \
    --meta_file $BUNDLE/configs/metadata.json \
    --logging_file $BUNDLE/configs/logging.conf \
    --config_file "$CONFIG" \
    --bundle_root $BUNDLE \
    $@
