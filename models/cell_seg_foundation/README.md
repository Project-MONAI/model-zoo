<!--
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Overview

The **VISTA2D** is a cell segmentation training and inference pipeline for cell imaging [[`Blog`](https://developer.nvidia.com/blog/advancing-cell-segmentation-and-morphology-analysis-with-nvidia-ai-foundation-model-vista-2d/)].

A pretrained model was trained on collection of 15K public microscopy images. The data collection and training can be reproduced following the [tutorial](./download_preprocessor/). Alternatively, the model can be retrained on your own dataset. The pretrained vista2d model achieves good performance on diverse set of cell types, microscopy image modalities, and can be further finetuned if necessary.  The codebase utilizes several components from other great works including [SegmentAnything](https://github.com/facebookresearch/segment-anything)  and [Cellpose](https://www.cellpose.org/), which must be pip installed as dependencies.  Vista2D codebase follows MONAI bundle format and its [specifications](https://docs.monai.io/en/stable/mb_specification.html).

<div align="center"> <img src="https://developer-blogs.nvidia.com/wp-content/uploads/2024/04/magnified-cells-1.png" width="800"/> </div>


### Model highlights

- Robust deep learning algorithm based on transformers
- Generalist model as compared to specialist models
- Multiple dataset sources and file formats supported
- Multiple modalities of imaging data collectively supported 
- Multi-GPU and multinode training support


### Generalization performance

Evaluation was performed for the VISTA2D model with multiple public datasets, such as TissueNet, LIVECell, Omnipose, DeepBacs, Cellpose, and [more](./docs/data_license.txt). A total of ~15K annotated cell images were collected to train the generalist VISTA2D model. This ensured broad coverage of many different types of cells, which were acquired by various imaging acquisition types. The benchmark results of the experiment were performed on held-out test sets for each public dataset that were already defined by the dataset contributors. Average precision at an IoU threshold of 0.5 was used for evaluating performance. The benchmark results are reported in comparison with the best numbers found in the literature, in addition to a specialist VISTA2D model trained only on a particular dataset or a subset of data. 

<div align="center"> <img src="https://developer-blogs.nvidia.com/wp-content/uploads/2024/04/vista-2d-model-precision-versus-specialist-model-baseline-performance.png" width="800"/> </div>



### Install dependencies

```
pip install monai fire tifffile imagecodecs pillow fastremap
pip install --no-deps cellpose natsort roifile
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install mlflow psutil pynvml #optional for MLFlow support
```

### Execute training
```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml
```

#### Quick run with a few data points
```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --quick True --train#trainer#max_epochs 3
```

### Execute multi-GPU training
```bash
torchrun --nproc_per_node=gpu -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml
```

### Execute validation
```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --pretrained_ckpt_name model.pt --mode eval
```
(can append `--quick True` for quick demoing)

### Execute multi-GPU validation
```bash
torchrun --nproc_per_node=gpu -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --mode eval
```

### Execute inference
```bash
python -m monai.bundle run --config_file configs/inference.json
```

### Execute multi-GPU inference
```bash
torchrun --nproc_per_node=gpu -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --mode infer --pretrained_ckpt_name model.pt
```
(can append `--quick True` for quick demoing)



#### Finetune starting from a trained checkpoint
(we use a smaller learning rate, small number of epochs, and initialize from a checkpoint)
```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --learning_rate=0.001 --train#trainer#max_epochs 20 --pretrained_ckpt_path /path/to/saved/model.pt
```


#### Configuration options

To disable the segmentation writing:
```
--postprocessing []
```

Load a checkpoint for validation or inference (relative path within results directory):
```
--pretrained_ckpt_name "model.pt"
```

Load a checkpoint for validation or inference (absolute path):
```
--pretrained_ckpt_path "/path/to/another/location/model.pt"
```

`--mode eval` or `--mode infer`will use the corresponding configurations from the `validate` or `infer`
of the `configs/hyper_parameters.yaml`.

By default the generated `model.pt` corresponds to the checkpoint at the best validation score,
`model_final.pt` is the checkpoint after the latest training epoch.


### Development

For development purposes it's possible to run the script directly (without monai bundle calls)

```bash
python scripts/workflow.py --config_file configs/hyper_parameters.yaml ...
torchrun --nproc_per_node=gpu -m  scripts/workflow.py --config_file configs/hyper_parameters.yaml  ..
```

### MLFlow support

Enable MLFlow logging by specifying "mlflow_tracking_uri" (can be local or remote URL).

```bash
python -m monai.bundle run_workflow "scripts.workflow.VistaCell" --config_file configs/hyper_parameters.yaml --mlflow_tracking_uri=http://127.0.0.1:8080
```

Optionally use "--mlflow_run_name=.." to specify MLFlow experiment name, and "--mlflow_log_system_metrics=True/False" to enable logging of CPU/GPU resources (requires pip install psutil pynvml)



### Unit tests

Test single GPU training:
```
python unit_tests/test_vista2d.py
```

Test multi-GPU training (may need to uncomment the `"--standalone"` in the `unit_tests/utils.py` file):
```
python unit_tests/test_vista2d_mgpu.py
```

## Compute Requirements
Min GPU memory requirements 16Gb.


## Contributing
Vista2D codebase follows MONAI bundle format and its [specifications](https://docs.monai.io/en/stable/mb_specification.html).
Make sure to run pre-commit before committing code changes to git
```bash
pip install pre-commit
python3 -m pre_commit run --all-files
```


## Community

Join the conversation on Twitter [@ProjectMONAI](https://twitter.com/ProjectMONAI) or join
our [Slack channel](https://projectmonai.slack.com/archives/C031QRE0M1C).

Ask and answer questions on [MONAI VISTA's GitHub discussions tab](https://github.com/Project-MONAI/VISTA/discussions).

## License

The codebase is under Apache 2.0 Licence. The model weight is released under CC-BY-NC-SA-4.0. For various public data licenses please see [data_license.txt](./docs/data_license.txt).

## Acknowledgement
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [Cellpose](https://www.cellpose.org/)