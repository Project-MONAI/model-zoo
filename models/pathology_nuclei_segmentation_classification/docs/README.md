# Model Overview
A pre-trained model for simultaneous segmentation and classification of nuclei within multi-tissue histology images based on CoNSeP data. The details of the model can be found in [1].

## Workflow
The model is trained to simultaneously segment and classify nuclei. Training is done via a two-stage approach. First initialized the model with pre-trained weights on the [ImageNet dataset](https://ieeexplore.ieee.org/document/5206848), trained only the decoders for the first 50 epochs, and then fine-tuned all layers for another 50 epochs. There are two training modes in total. If "original" mode is specified, [270, 270] and [80, 80] are used for `patch_size` and `out_size` respectively. If "fast" mode is specified, [256, 256] and [164, 164] are used for `patch_size` and `out_size` respectively. The results shown below are based on the "fast" mode.

### Pre-trained weights
The first stage is trained with pre-trained weights from some internal data.The [original author's repo](https://github.com/vqdang/hover_net#data-format) also provides pre-trained weights but for non-commercial use.
Each user is responsible for checking the content of models/datasets and the applicable licenses and determining if suitable for the intended use.

`PRETRAIN_MODEL_URL` is "https://drive.google.com/u/1/uc?id=1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5&export=download" which can be used in bash code below.

![Model workflow](https://developer.download.nvidia.com/assets/Clara/Images/monai_hovernet_pipeline.png)

## Data
The training data is from <https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/>.

- Target: segment instance-level nuclei and classify the nuclei type
- Task: Segmentation and classification
- Modality: RGB images
- Size: 41 image tiles (2009 patches)

The provided labelled data was partitioned, based on the original split, into training (27 tiles) and testing (14 tiles) datasets.

After download the datasets, please run `scripts/prepare_patches.py` to prepare patches from tiles. Prepared patches are saved in `your-concep-dataset-path`/Prepared. The implementation is referring to <https://github.com/vqdang/hover_net/blob/master/extract_patches.py>. The command is like:

```
python scripts/prepare_patches.py -root your-concep-dataset-path
```

## Training configuration
This model utilized a two-stage approach. The training was performed with the following:

- GPU: At least 24GB of GPU memory.
- Actual Model Input: 256 x 256
- AMP: True
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss: HoVerNetLoss

## Input
Input: RGB images

## Output
Output: a dictionary with the following keys:

1. nucleus_prediction: predict whether or not a pixel belongs to the nuclei or background
2. horizontal_vertical: predict the horizontal and vertical distances of nuclear pixels to their centres of mass
3. type_prediction: predict the type of nucleus for each pixel

## Performance
The achieved metrics on the validation data are:

Fast mode:
- Binary Dice: 0.8291
- PQ: 0.4973
- F1d: 0.7417

Note: Binary Dice is calculated based on the whole input. PQ and F1d were calculated from https://github.com/vqdang/hover_net#inference.

Please note that this bundle is non-deterministic because of the bilinear interpolation used in the network. Therefore, reproducing the training process may not get exactly the same performance.
Please refer to https://pytorch.org/docs/stable/notes/randomness.html#reproducibility for more details about reproducibility.

#### Training Loss and Dice
stage1:
![A graph showing the training loss and the mean dice over 50 epochs in stage1](https://developer.download.nvidia.com/assets/Clara/Images/monai_pathology_segmentation_classification_train_stage0_v2.png)

stage2:
![A graph showing the training loss and the mean dice over 50 epochs in stage2](https://developer.download.nvidia.com/assets/Clara/Images/monai_pathology_segmentation_classification_train_stage1_v2.png)

#### Validation Dice
stage1:

![A graph showing the validation mean dice over 50 epochs in stage1](https://developer.download.nvidia.com/assets/Clara/Images/monai_pathology_segmentation_classification_val_stage0_v2.png)

stage2:

![A graph showing the validation mean dice over 50 epochs in stage2](https://developer.download.nvidia.com/assets/Clara/Images/monai_pathology_segmentation_classification_val_stage1_v2.png)

## MONAI Bundle Commands
In addition to the Pythonic APIs, a few command line interfaces (CLI) are provided to interact with the bundle. The CLI supports flexible use cases, such as overriding configs at runtime and predefining arguments in a file.

For more details usage instructions, visit the [MONAI Bundle Configuration Page](https://docs.monai.io/en/latest/config_syntax.html).

#### Execute training, the evaluation in the training were evaluated on patches:

- Run first stage
```
python -m monai.bundle run --config_file configs/train.json --network_def#pretrained_url `PRETRAIN_MODEL_URL` --stage 0
```

- Run second stage
```
python -m monai.bundle run --config_file configs/train.json --network_def#freeze_encoder False --network_def#pretrained_url None --stage 1
```

#### Override the `train` config to execute multi-GPU training:

- Run first stage
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run --config_file "['configs/train.json','configs/multi_gpu_train.json']" --batch_size 8 --network_def#freeze_encoder True --network_def#pretrained_url `PRETRAIN_MODEL_URL --stage 0
```

- Run second stage
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run --config_file "['configs/train.json','configs/multi_gpu_train.json']" --batch_size 4 --network_def#freeze_encoder False --network_def#pretrained_url None --stage 1
```

#### Override the `train` config to execute evaluation with the trained model, here we evaluated dice from the whole input instead of the patches:

```
python -m monai.bundle run --config_file "['configs/train.json','configs/evaluate.json']"
```

#### Execute inference:

```
python -m monai.bundle run --config_file configs/inference.json
```

# Disclaimer
This is an example, not to be used for diagnostic purposes.

# References
[1] Simon Graham, Quoc Dang Vu, Shan E Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, Nasir Rajpoot, Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images, Medical Image Analysis, 2019 https://doi.org/10.1016/j.media.2019.101563

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
