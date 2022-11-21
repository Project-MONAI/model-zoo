#!/usr/bin/env bash

LEVEL=6
SPACING=0.243
READER=openslide
EVAL_DIR=../eval
GROUND_TRUTH_DIR=/workspace/data/medical/pathology/testing/ground_truths

echo "=> Level= ${LEVEL}"
echo "=> Spacing = ${SPACING}"
echo "=> WSI Reader: ${READER}"
echo "=> Evaluation output directory: ${EVAL_DIR}"
echo "=> Ground truth directory: ${GROUND_TRUTH_DIR}"

python3 ./lesion_froc.py \
    --level $LEVEL \
    --spacing $SPACING \
    --reader $READER \
    --eval-dir ${EVAL_DIR} \
    --ground-truth-dir ${GROUND_TRUTH_DIR}
