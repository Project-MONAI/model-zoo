# Description
A pre-trained model for the endoscopic inbody classification task.

# Model Overview
This model is trained using the SEResNet50 structure, whose details can be found in [1]. All datasets are from private samples of [Activ Surgical](https://www.activsurgical.com/). Samples in training and validation dataset are from the same 4 videos, while test samples are from different two videos.
The [pytorch model](https://drive.google.com/file/d/14CS-s1uv2q6WedYQGeFbZeEWIkoyNa-x/view?usp=sharing) and [torchscript model](https://drive.google.com/file/d/1fOoJ4n5DWKHrt9QXTZ2sXwr9C-YvVGCM/view?usp=sharing) are shared in google drive. Modify the `bundle_root` parameter specified in `configs/train.json` and `configs/inference.json` to reflect where models are downloaded. Expected directory path to place downloaded models is `models/` under `bundle_root`.

## Data
Datasets used in this work were provided by [Activ Surgical](https://www.activsurgical.com/). Here is a [link](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/inbody_outbody_samples.zip) of 20 samples (10 in-body and 10 out-body) to show what this dataset looks like. After downloading this dataset, python script in `scripts` folder naming `data_process` can be used to get label json files by running the command below and replacing datapath and outpath parameters.
```
python scripts/data_process.py --datapath /path/to/data/root --outpath /path/to/label/folder
```

After generating label files, please modify the `dataset_dir` parameter specified in `configs/train.json` and `configs/inference.json` to reflect where label files are.

The input label json should be a list made up by dicts which includes `image` and `label` keys. An example format is shown below.

```
[
    {
        "image":"/path/to/image/image_name0.jpg",
        "label": 0
    },
    {
        "image":"/path/to/image/image_name1.jpg",
        "label": 0
    },
    {
        "image":"/path/to/image/image_name2.jpg",
        "label": 1
    },
    ....
    {
        "image":"/path/to/image/image_namek.jpg",
        "label": 0
    },
]
```

## Training configuration
The training was performed with an at least 12GB-memory GPU.

Actual Model Input: 256 x 256 x 3

## Input and output formats
Input: 3 channel video frames

Output: probability vector whose length equals to 2: Label 0: in body; Label 1: out body

## Scores
This model achieves the following accuracy score on the test dataset:

Accuracy = 0.98

## commands example
Execute training:

```
python -m monai.bundle run training \
    --meta_file configs/metadata.json \
    --config_file configs/train.json \
    --logging_file configs/logging.conf
```

Override the `train` config to execute multi-GPU training:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run training \
    --meta_file configs/metadata.json \
    --config_file "['configs/train.json','configs/multi_gpu_train.json']" \
    --logging_file configs/logging.conf
```

Please note that the distributed training related options depend on the actual running environment, thus you may need to remove `--standalone`, modify `--nnodes` or do some other necessary changes according to the machine you used.
Please refer to [pytorch's official tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for more details.

Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run evaluating \
    --meta_file configs/metadata.json \
    --config_file "['configs/train.json','configs/evaluate.json']" \
    --logging_file configs/logging.conf
```

Execute inference:

```
python -m monai.bundle run evaluating \
    --meta_file configs/metadata.json \
    --config_file configs/inference.json \
    --logging_file configs/logging.conf
```

Export checkpoint to TorchScript file:

```
python -m monai.bundle ckpt_export network_def \
    --filepath models/model.ts \
    --ckpt_file models/model.pt \
    --meta_file configs/metadata.json \
    --config_file configs/inference.json
```

Export checkpoint to onnx file, which has been tested on pytorch 1.12.0:

```
python scripts/export_to_onnx.py --model models/model.pt --outpath models/model.onnx
```

Export TensorRT float16 model from the onnx model:

```
trtexec --onnx=models/model.onnx --saveEngine=models/model.trt --fp16 \
    --minShapes=INPUT__0:1x3x256x256 \
    --optShapes=INPUT__0:16x3x256x256 \
    --maxShapes=INPUT__0:32x3x256x256 \
    --shapes=INPUT__0:8x3x256x256
```
This command need TensorRT with correct CUDA installed in the environment. For the detail of installing TensorRT, please refer to [this link](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

# References
[1] J. Hu, L. Shen and G. Sun, Squeeze-and-Excitation Networks, 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 7132-7141. https://arxiv.org/pdf/1709.01507.pdf

# License
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
