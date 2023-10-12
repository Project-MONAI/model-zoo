#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate monailabel

export CUDA_VISIBLE_DEVICES=0

homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

BUNDLE="$(cd "$homedir/.." && pwd)"

LABELDIR="$BUNDLE/monailabel"

BUNDLENAME=$(basename "$BUNDLE")

if [ ! -d "$LABELDIR" ]
then
    mkdir "$LABELDIR"
    mkdir "$LABELDIR/datasets"
    cd "$LABELDIR"
    monailabel apps --download --name monaibundle
    mkdir "$LABELDIR/monaibundle/model"
    cd "$LABELDIR/monaibundle/model"
    ln -s "$BUNDLE" $BUNDLENAME
fi

cd "$LABELDIR"
monailabel start_server --app monaibundle --studies datasets --conf models $BUNDLENAME $*
