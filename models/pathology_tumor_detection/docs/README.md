# Model Overview

A pre-trained model for automated detection of metastases in whole-slide histopathology images.

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

### Preprocessing

This bundle expects the training/validation data (whole slide images) reside in a `{dataset_dir}/training/images`. By default `dataset_dir` is pointing to `/workspace/data/medical/pathology/` You can modify `dataset_dir` in the bundle config files to point to a different directory.

To reduce the computation burden during the inference, patches are extracted only where there is tissue and ignoring the background according to a tissue mask. Please also create a directory for prediction output. By default `output_dir` is set to `eval` folder under the bundle root.

Please refer to "Annotation" section of [Camelyon challenge](https://camelyon17.grand-challenge.org/Data/) to prepare ground truth images, which are needed for FROC computation. By default, this data set is expected to be at `/workspace/data/medical/pathology/ground_truths`. But it can be modified in `evaluate_froc.sh`.

## Training configuration

The training was performed with the following:

- Config file: train.config
- GPU: at least 16 GB of GPU memory.
- Actual Model Input: 224 x 224 x 3
- AMP: True
- Optimizer: Novograd
- Learning Rate: 1e-3
- Loss: BCEWithLogitsLoss
- Whole slide image reader: cuCIM (if running on Windows or Mac, please install `OpenSlide` on your system and change `wsi_reader` to "OpenSlide")

### Input

The training pipeline is a json file (dataset.json) which includes path to each WSI, the location and the label information for each training patch.

### Output

A probability number of the input patch being tumor or normal.

### Inference on a WSI

Inference is performed on WSI in a sliding window manner with specified stride. A foreground mask is needed to specify the region where the inference will be performed on, given that background region which contains no tissue at all can occupy a significant portion of a WSI. Output of the inference pipeline is a probability map of size 1/stride of original WSI size.

## Performance

FROC score is used for evaluating the performance of the model. After inference is done, `evaluate_froc.sh` needs to be run to evaluate FROC score based on predicted probability map (output of inference) and the ground truth tumor masks.
This model achieve the 0.91 accuracy on validation patches, and FROC of 0.72 on the 48 Camelyon testing data that have ground truth annotations available.

![A Graph showing Train Acc, Train Loss, and Validation Acc](https://developer.download.nvidia.com/assets/Clara/Images/monai_pathology_tumor_detection_train_and_val_metrics_v3.png)

The `pathology_tumor_detection` bundle supports the TensorRT acceleration. The table below shows the speedup ratios benchmarked on an A100 80G GPU, in which the `model computation` means the speedup ratio of model's inference with a random input without preprocessing and postprocessing and the `end2end` means run the bundle end to end with the TensorRT based model. The `torch_fp32` and `torch_amp` is for the pytorch model with or without `amp` mode. The `trt_fp32` and `trt_fp16` is for the TensorRT based model converted in corresponding precision. The `speedup amp`, `speedup fp32` and `speedup fp16` is the speedup ratio of corresponding models versus the pytorch float32 model, while the `amp vs fp16` is between the pytorch amp model and the TensorRT float16 based model.

Please notice that the benchmark results are tested on one WSI image since the images are too large to benchmark. And the inference time in the end2end line stands for one patch of the whole image.

| method | torch_fp32(ms) | torch_amp(ms) | trt_fp32(ms) | trt_fp16(ms) | speedup amp | speedup fp32 | speedup fp16 | amp vs fp16|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| model computation |1.93 | 2.52 | 1.61 | 1.33 | 0.77 | 1.20 | 1.45 | 1.89 |
| end2end |224.97 | 223.50 | 222.65 | 224.03 | 1.01 | 1.01 | 1.00 | 1.00 |

## MONAI Bundle Commands

In addition to the Pythonic APIs, a few command line interfaces (CLI) are provided to interact with the bundle. The CLI supports flexible use cases, such as overriding configs at runtime and predefining arguments in a file.

For more details usage instructions, visit the [MONAI Bundle Configuration Page](https://docs.monai.io/en/latest/config_syntax.html).

#### Execute training

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf
```

#### Override the `train` config to execute multi-GPU training

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run training --meta_file configs/metadata.json --config_file "['configs/train.json','configs/multi_gpu_train.json']" --logging_file configs/logging.conf
```

Please note that the distributed training-related options depend on the actual running environment; thus, users may need to remove `--standalone`, modify `--nnodes`, or do some other necessary changes according to the machine used. For more details, please refer to [pytorch's official tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

#### Execute inference

```
CUDA_LAUNCH_BLOCKING=1 python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf
```

#### Evaluate FROC metric

```
cd scripts && source evaluate_froc.sh
```

#### Export checkpoint to TorchScript file

```
python -m monai.bundle ckpt_export network_def --filepath models/model.ts --ckpt_file models/model.pt --meta_file configs/metadata.json --config_file configs/inference.json
```

#### Export checkpoint to TensorRT based models with fp32 and fp16 precision:

```
python -m monai.bundle trt_export --net_id network_def --filepath models/model_trt.ts --ckpt_file models/model.pt --meta_file configs/metadata.json --config_file configs/inference.json --precision <fp32/fp16> --dynamic_batchsize [[1, 400, 600]]
```

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
