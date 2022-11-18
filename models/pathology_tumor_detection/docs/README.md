# Model Overview

A pre-trained model for automated detection of metastases in whole-slide histopathology images.

## Workflow

The model is trained based on ResNet18 [1] with the last fully connected layer replaced by a 1x1 convolution layer.
![Diagram showing the flow from model input, through the model architecture, and to model output](http://developer.download.nvidia.com/assets/Clara/Images/clara_pt_pathology_metastasis_detection_workflow.png)

## Data

All the data used to train, validate, and test this model is from [Camelyon-16 Challenge](https://camelyon16.grand-challenge.org/). You can download all the images for "CAMELYON16" data set from various sources listed [here](https://camelyon17.grand-challenge.org/Data/).

Location information for training/validation patches (the location on the whole slide image where patches are extracted) are adopted from [NCRF/coords](https://github.com/baidu-research/NCRF/tree/master/coords).

Annotation information are adopted from [NCRF/jsons](https://github.com/baidu-research/NCRF/tree/master/jsons).

- Target: Tumor
- Task: Detection
- Modality: Histopathology
- Size: 270 WSIs for training/validation, 48 WSIs for testing

### Data Preparation

This bundle expects the training/validation data (whole slide images) reside in a `{data_root}/training/images`. By default `data_root` is pointing to `/workspace/data/medical/pathology/` You can modify `data_root` in the bundle config files to point to a different directory.

To reduce the computation burden during the inference, patches are extracted only where there is tissue and ignoring the background according to a tissue mask. Please also create a directory for prediction output. By default `output_dir` is set to `eval` folder under the bundle root.

Please refer to "Annotation" section of [Camelyon challenge](https://camelyon17.grand-challenge.org/Data/) to prepare ground truth images, which are needed for FROC computation. By default, this data set is expected to be at `/workspace/data/medical/pathology/ground_truths`. But it can be modified in `evaluate_froc.sh`.

# Training configuration

The training was performed with the following:

- Config file: train.config
- GPU: at least 16 GB of GPU memory.
- Actual Model Input: 224 x 224 x 3
- AMP: True
- Optimizer: Novograd
- Learning Rate: 1e-3
- Loss: BCEWithLogitsLoss
- Whole slide image reader: cuCIM (if running on Windows or Mac, please install `OpenSlide` on your system and change `wsi_reader` to "OpenSlide")

## Input

Input: Input for the training pipeline is a json file (dataset.json) which includes path to each WSI, the location and the label information for each training patch.

1. Extract 224 x 224 x 3 patch from WSI according to the location information from json
2. Randomly applying color jittering
3. Randomly applying spatial flipping
4. Randomly applying spatial rotation
5. Randomly applying spatial zooming
6. Randomly applying intensity scaling

## Output

Output of the network is a probability number of the input patch being tumor or normal.

## Inference on a WSI

Inference is performed on WSI in a sliding window manner with specified stride. A foreground mask is needed to specify the region where the inference will be performed on, given that background region which contains no tissue at all can occupy a significant portion of a WSI. Output of the inference pipeline is a probability map of size 1/stride of original WSI size.

# Model Performance

FROC score is used for evaluating the performance of the model. After inference is done, `evaluate_froc.sh` needs to be run to evaluate FROC score based on predicted probability map (output of inference) and the ground truth tumor masks.
This model achieve the ~0.92 accuracy on validation patches, and FROC of ~0.72 on the 48 Camelyon testing data that have ground truth annotations available.

# Commands example

Execute training:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf
```

Override the `train` config to execute multi-GPU training:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run training --meta_file configs/metadata.json --config_file "['configs/train.json','configs/multi_gpu_train.json']" --logging_file configs/logging.conf
```

Please note that the distributed training related options depend on the actual running environment, thus you may need to remove `--standalone`, modify `--nnodes` or do some other necessary changes according to the machine you used.
Please refer to [pytorch's official tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for more details.

Execute inference:

```
CUDA_LAUNCH_BLOCKING=1 python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf
```

Export checkpoint to TorchScript file:

TorchScript conversion is currently not supported.

# References

[1] He, Kaiming, et al, "Deep Residual Learning for Image Recognition." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016. <https://arxiv.org/pdf/1512.03385.pdf>

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
