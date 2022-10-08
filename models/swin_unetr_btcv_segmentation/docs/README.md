# Description
A pre-trained model for volumetric (3D) multi-organ segmentation from CT image.

# Model Overview
A pre-trained Swin UNETR [1,2] for volumetric (3D) multi-organ segmentation using CT images from Beyond the Cranial Vault (BTCV) Segmentation Challenge dataset [3].
## Data
The training data is from the [BTCV dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480/) (Please regist in `Synapse` and download the `Abdomen/RawData.zip`).
The dataset format needs to be redefined using the following commands:

```
unzip RawData.zip
mv RawData/Training/img/ RawData/imagesTr
mv RawData/Training/label/ RawData/labelsTr
mv RawData/Testing/img/ RawData/imagesTs
```

- Target: Multi-organs
- Task: Segmentation
- Modality: CT
- Size: 30 3D volumes (24 Training + 6 Testing)

## Training configuration
The training was performed with at least 32GB-memory GPUs.

Actual Model Input: 96 x 96 x 96

## Input and output formats
Input: 1 channel CT image

Output: 14 channels: 0:Background, 1:Spleen, 2:Right Kidney, 3:Left Kideny, 4:Gallbladder, 5:Esophagus, 6:Liver, 7:Stomach, 8:Aorta, 9:IVC, 10:Portal and Splenic Veins, 11:Pancreas, 12:Right adrenal gland, 13:Left adrenal gland

## Performance
A graph showing the validation mean Dice for 5000 epochs.

![](./val_dice.png) <br>

This model achieves the following Dice score on the validation data (our own split from the training dataset):

Mean Dice = 0.8283

Note that mean dice is computed in the original spacing of the input data.
## commands example
Execute training:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf
```

Override the `train` config to execute multi-GPU training:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run training --meta_file configs/metadata.json --config_file "['configs/train.json','configs/multi_gpu_train.json']" --logging_file configs/logging.conf
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

TorchScript conversion is currently not supported.

# Disclaimer
This is an example, not to be used for diagnostic purposes.

# References
[1] Hatamizadeh, Ali, et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images." arXiv preprint arXiv:2201.01266 (2022). https://arxiv.org/abs/2201.01266.

[2] Tang, Yucheng, et al. "Self-supervised pre-training of swin transformers for 3d medical image analysis." arXiv preprint arXiv:2111.14791 (2021). https://arxiv.org/abs/2111.14791.

[3] Landman B, et al. "MICCAI multi-atlas labeling beyond the cranial vault–workshop and challenge." In Proc. of the MICCAI Multi-Atlas Labeling Beyond Cranial Vault—Workshop Challenge 2015 Oct (Vol. 5, p. 12).

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
