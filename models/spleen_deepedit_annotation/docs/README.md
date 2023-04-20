# Model Overview
A pre-trained model for 3D segmentation of the spleen organ from CT images using DeepEdit.

DeepEdit is an algorithm that combines the power of two models in one single architecture. It allows the user to perform inference as a standard segmentation method (i.e., UNet) and interactively segment part of an image using clicks [2]. DeepEdit aims to facilitate the user experience and, at the same time, develop new active learning techniques.

The model was trained on 32 images and validated on 9 images.

## Data
The training dataset is the Spleen Task from the Medical Segmentation Decathalon. Users can find more details on the datasets at http://medicaldecathlon.com/.

- Target: Spleen
- Modality: CT
- Size: 61 3D volumes (41 Training + 20 Testing)
- Source: Memorial Sloan Kettering Cancer Center
- Challenge: Large-ranging foreground size

## Training configuration
The training as performed with the following:
- GPU: at least 12GB of GPU memory
- Actual Model Input: 128 x 128 x 128
- AMP: True
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss: DiceCELoss

### Input
Three channels
- CT image
- Spleen Segment
- Background Segment

### Output
Two channels
- Label 1: spleen
- Label 0: everything else

## Performance

Dice score is used for evaluating the performance of the model. This model achieves a dice score of greater than 0.90, depending on the number of simulated clicks.

#### Training Dice
![A graph showing the train dice over 90 epochs.](https://developer.download.nvidia.com/assets/Clara/Images/monai_spleen_deepedit_annotation_train_dice_v2.png)

#### Training Loss
![A graph showing the training loss over 90 epochs.](https://developer.download.nvidia.com/assets/Clara/Images/monai_spleen_deepedit_annotation_train_loss_v2.png)

#### Validation Dice
![A graph showing the validation dice over 90 epochs.](https://developer.download.nvidia.com/assets/Clara/Images/monai_spleen_deepedit_annotation_val_dice_v2.png)

## MONAI Bundle Commands
In addition to the Pythonic APIs, a few command line interfaces (CLI) are provided to interact with the bundle. The CLI supports flexible use cases, such as overriding configs at runtime and predefining arguments in a file.

For more details usage instructions, visit the [MONAI Bundle Configuration Page](https://docs.monai.io/en/latest/config_syntax.html).

#### Execute training:

```
python -m monai.bundle run --config_file configs/train.json
```

#### Override the `train` config to execute multi-GPU training:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run --config_file "['configs/train.json','configs/multi_gpu_train.json']"
```

Please note that the distributed training-related options depend on the actual running environment; thus, users may need to remove `--standalone`, modify `--nnodes`, or do some other necessary changes according to the machine used. For more details, please refer to [pytorch's official tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

####  Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run --config_file "['configs/train.json','configs/evaluate.json']"
```

####  Execute inference:

```
python -m monai.bundle run --config_file configs/inference.json
```

# References
[1] Diaz-Pinto, Andres, et al. DeepEdit: Deep Editable Learning for Interactive Segmentation of 3D Medical Images. MICCAI Workshop on Data Augmentation, Labelling, and Imperfections. MICCAI 2022.

[2] Diaz-Pinto, Andres, et al. "MONAI Label: A framework for AI-assisted Interactive Labeling of 3D Medical Images." arXiv preprint arXiv:2203.12362 (2022).

[3] Sakinis, Tomas, et al. "Interactive segmentation of medical images through fully convolutional neural networks." arXiv preprint arXiv:1903.08205 (2019).

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
