#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate monai

homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

BUNDLE="$(cd "$homedir/.." && pwd)"

echo "Bundle root: $BUNDLE"

export PYTHONPATH="$BUNDLE"

python -m monai.bundle run \
    --meta_file "$BUNDLE/configs/metadata.json" \
    --config_file "$BUNDLE/configs/train.yaml" \
    --bundle_root "$BUNDLE" \
    $@
