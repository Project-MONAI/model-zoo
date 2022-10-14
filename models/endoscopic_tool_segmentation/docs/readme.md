# Description
A pre-trained model for the endoscopic tool segmentation task.

# Model Overview
This model is trained using a flexible unet structure with an efficient-b0 [1] as the backbone and a UNet architecture [2] as the decoder. Datasets use private samples from [Activ Surgical](https://www.activsurgical.com/).
The [pytorch model](https://drive.google.com/file/d/14r6WmzaZrgaWLGu0O9vSAzdeIGVFQ3cs/view?usp=sharing) and [torchscript model](https://drive.google.com/file/d/1i-e5xXHtmvmqitwUP8Q3JqvnmN3mlrEm/view?usp=sharing) are shared in google drive. Details can be found in large_files.yml file. Modify the "bundle_root" parameter specified in configs/train.json and configs/inference.json to reflect where models are downloaded. Expected directory path to place downloaded models is "models/" under "bundle_root".

## Data
Datasets used in this work were provided by [Activ Surgical](https://www.activsurgical.com/).

Since datasets are private, existing public datasets like [EndoVis 2017](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Data/) can be used to train a similar model.

When using EndoVis or any other dataset, it should be divided into "train", "valid" and "test" folders. Samples in each folder would better be images and converted to jpg format. Otherwise, "images", "labels", "val_images" and "val_labels" parameters in "configs/train.json" and "datalist" in "configs/inference.json" should be modified to fit given dataset. After that, "dataset_dir" parameter in "configs/train.json" and "configs/inference.json" should be changed to root folder which contains previous "train", "valid" and "test" folders.

Please notice that loading data operation in this bundle is adaptive. If images and labels are not in the same format, it may lead to a mismatching problem. For example, if images are in jpg format and labels are in npy format, PIL and Numpy readers will be used separately to load images and labels. Since these two readers have their own way to parse file's shape, loaded labels will be transpose of the correct ones and incur a missmatching problem.

## Training configuration
The training was performed with an at least 12GB-memory GPU.

Actual Model Input: 736 x 480 x 3

## Input and output formats
Input: 3 channel video frames

Output: 2 channels: Label 1: tools; Label 0: everything else

## Scores
This model achieves the following IoU score on the test dataset (our own split from the first batch data):

Mean IoU = 0.87

## commands example
Execute training:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf
```

Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.json','configs/evaluate.json']" --logging_file configs/logging.conf
```

Execute inference:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf
```

Export checkpoint to TorchScript file:

```
python -m monai.bundle ckpt_export network_def --filepath models/model.ts --ckpt_file models/model.pt --meta_file configs/metadata.json --config_file configs/inference.json
```

Export checkpoint to onnx file, which has been tested on pytorch 1.12.0:

```
python scripts/export_to_onnx.py --model models/model.pt --outpath models/model.onnx
```

Export TorchScript file to a torchscript module targeting a TensorRT engine with float16 precision.

```
torchtrtc -p f16 models/model.ts models/model_trt.ts "[(1,3,736,480);(4,3,736,480);(8,3,736,480)]"
```
The last parameter is the dynamic input shape in which each parameter means "[(MIN_BATCH, MIN_CHANNEL, MIN_WIDTH, MIN_HEIGHT), (OPT_BATCH, .., ..., OPT_HEIGHT), (MAX_BATCH, .., ..., MAX_HEIGHT)]". Please notice if using docker, the TensorRT CUDA must match the environment CUDA and the Torch-TensorRT c++&python version must be installed. For more examples on how to use the Torch-TensorRT, you can go to this [link](https://pytorch.org/TensorRT/). The [github source code link](https://github.com/pytorch/TensorRT) here shows the detail about how to install it on your own environment.

Export TensorRT float16 model from the onnx model:

```
polygraphy surgeon sanitize --fold-constants models/model.onnx -o models/new_model.onnx
```

```
trtexec --onnx=models/new_model.onnx --saveEngine=models/model.trt --fp16 --minShapes=INPUT__0:1x3x736x480 --optShapes=INPUT__0:4x3x736x480 --maxShapes=INPUT__0:8x3x736x480 --shapes=INPUT__0:4x3x736x480
```
This command need TensorRT with correct CUDA installed in the environment. For the detail of installing TensorRT, please refer to [this link](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html). In addition, there are padding operations in this FlexibleUNet structure that not support by TensorRT. Therefore, when tried to convert the onnx model to a TensorRT engine, an extra polygraphy command is needed to execute.


# References
[1] Tan, M. and Le, Q. V. Efficientnet: Rethinking model scaling for convolutional neural networks. ICML, 2019a. https://arxiv.org/pdf/1905.11946.pdf

[2] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015. https://arxiv.org/pdf/1505.04597.pdf

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
