# Description
A pre-trained model for volumetric (3D) segmentation of the spleen from CT image.

# Model Overview
This model is trained using the runner-up [1] awarded pipeline of the "Medical Segmentation Decathlon Challenge 2018" using the UNet architecture [2] with 32 training images and 9 validation images.

![image](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_spleen_ct_segmentation_workflow.png) 

## Data
The training dataset is Task09_Spleen.tar from http://medicaldecathlon.com/.

## Training configuration
The segmentation of spleen region is formulated as the voxel-wise binary classification. Each voxel is predicted as either foreground (spleen) or background. And the model is optimized with gradient descent method minimizing Dice + cross entropy loss between the predicted mask and ground truth segmentation. 

The training was performed with the following:

- GPU: at least 12GB of GPU memory
- Actual Model Input: 96 x 96 x 96
- AMP: True
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss: DiceCELoss

Pre-processing transforms:

1. Convert data to channel-first
2. Resample to resolution 1.5 x 1.5 x 2 mm
3. Scale intensity
4. Cropping foreground surrounding regions
5. Cropping random fixed sized regions of size [96,96,96] with the center being a foreground or background voxel at ratio 1 : 1
6. Randomly shifting intensity of the volume

## Input and output formats
Input: 1 channel CT image

Output: 2 channels: Label 1: spleen; Label 0: everything else

## Scores
This model achieves the following Dice score on the validation data (our own split from the training dataset):

Mean Dice = 0.96

## Training Performance
A graph showing the training loss over 1260 epochs (10080 iterations).

![](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_spleen_ct_segmentation_train_2.png) <br>

## Validation Performance
A graph showing the validation mean Dice over 1260 epochs.

![](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_spleen_ct_segmentation_val_2.png) <br>


## commands example
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

Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.json','configs/evaluate.json']" --logging_file configs/logging.conf
```

Override the `train` config and `evaluate` config to execute multi-GPU evaluation:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.json','configs/evaluate.json','configs/multi_gpu_evaluate.json']" --logging_file configs/logging.conf
```

Execute inference:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf
```

# Disclaimer
This is an example, not to be used for diagnostic purposes.

# References
[1] Xia, Yingda, et al. "3D Semi-Supervised Learning with Uncertainty-Aware Multi-View Co-Training." arXiv preprint arXiv:1811.12506 (2018). https://arxiv.org/abs/1811.12506.

[2] Kerfoot E., Clough J., Oksuz I., Lee J., King A.P., Schnabel J.A. (2019) Left-Ventricle Quantification Using Residual U-Net. In: Pop M. et al. (eds) Statistical Atlases and Computational Models of the Heart. Atrial Segmentation and LV Quantification Challenges. STACOM 2018. Lecture Notes in Computer Science, vol 11395. Springer, Cham. https://doi.org/10.1007/978-3-030-12029-0_40

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
