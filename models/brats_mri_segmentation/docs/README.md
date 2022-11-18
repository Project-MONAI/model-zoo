# Model Overview
A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data. The whole pipeline is modified from [clara_pt_brain_mri_segmentation](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/med/models/clara_pt_brain_mri_segmentation).

## Workflow

The model is trained to segment 3 nested subregions of primary brain tumors (gliomas): the "enhancing tumor" (ET), the "tumor core" (TC), the "whole tumor" (WT) based on 4 aligned input MRI scans (T1c, T1, T2, FLAIR).
- The ET is described by areas that show hyper intensity in T1c when compared to T1, but also when compared to "healthy" white matter in T1c.
- The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (fluid-filled) and the non-enhancing (solid) parts of the tumor.
-  The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edema (ED), which is typically depicted by hyper-intense signal in FLAIR.

![](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_brain_mri_segmentation_workflow.png)

## Data

The training data is from the [Multimodal Brain Tumor Segmentation Challenge (BraTS) 2018](https://www.med.upenn.edu/cbica/sbia/brats2018/tasks.html).

- Target: 3 tumor subregions
- Task: Segmentation
- Modality: MRI
- Size: 285 3D volumes (4 channels each)

The provided labelled data was partitioned, based on our own split, into training (200 studies), validation (42 studies) and testing (43 studies) datasets.

Please run `scripts/prepare_datalist.py` to produce the data list. The command is like:

```
python scripts/prepare_datalist.py --path your-brats18-dataset-path
```

## Training configuration

This model utilized a similar approach described in 3D MRI brain tumor segmentation
using autoencoder regularization, which was a winning method in BraTS2018 [1]. The training was performed with the following:

- GPU: At least 16GB of GPU memory.
- Actual Model Input: 224 x 224 x 144
- AMP: True
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss: DiceLoss

## Input

Input: 4 channel MRI (4 aligned MRIs T1c, T1, T2, FLAIR at 1x1x1 mm)

1. Normalizing to unit std with zero mean
2. Randomly cropping to (224, 224, 144)
3. Randomly spatial flipping
4. Randomly scaling and shifting intensity of the volume

## Output

Output: 3 channels
- Label 0: TC tumor subregion
- Label 1: WT tumor subregion
- Label 2: ET tumor subregion

## Model Performance

The achieved Dice scores on the validation data are:
- Tumor core (TC): 0.8559
- Whole tumor (WT): 0.9026
- Enhancing tumor (ET): 0.7905
- Average: 0.8518

## commands example

Execute training:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf
```

Override the `train` config to execute multi-GPU training:

```
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m monai.bundle run training --meta_file configs/metadata.json --config_file "['configs/train.json','configs/multi_gpu_train.json']" --logging_file configs/logging.conf
```

Please note that the distributed training related options depend on the actual running environment, thus you may need to remove `--standalone`, modify `--nnodes` or do some other necessary changes according to the machine you used.
Please refer to [pytorch's official tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for more details.

Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.json','configs/evaluate.json']" --logging_file configs/logging.conf
```

Execute inference:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf
```

# Training
A graph showing the training loss and the mean dice over 300 epochs.
![](https://developer.download.nvidia.com/assets/Clara/Images/monai_brats_mri_segmentation_train.png)

# Validation
A graph showing the validation mean dice over 300 epochs. 
![](https://developer.download.nvidia.com/assets/Clara/Images/monai_brats_mri_segmentation_val.png)


# Disclaimer

This is an example, not to be used for diagnostic purposes.

# References

[1] Myronenko, Andriy. "3D MRI brain tumor segmentation using autoencoder regularization." International MICCAI Brainlesion Workshop. Springer, Cham, 2018. https://arxiv.org/abs/1810.11654.

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
