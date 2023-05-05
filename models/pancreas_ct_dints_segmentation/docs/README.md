# Model Overview
A neural architecture search algorithm for volumetric (3D) segmentation of the pancreas and pancreatic tumor from CT image. This model is trained using the neural network model from the neural architecture search algorithm, DiNTS [1].

![image](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_net_arch_search_segmentation_workflow_4-1.png)

## Data
The training dataset is the Panceas Task from the Medical Segmentation Decathalon. Users can find more details on the datasets at http://medicaldecathlon.com/.

- Target: Liver and tumour
- Modality: Portal venous phase CT
- Size: 420 3D volumes (282 Training +139 Testing)
- Source: Memorial Sloan Kettering Cancer Center
- Challenge: Label unbalance with large (background), medium (pancreas) and small (tumour) structures.

### Preprocessing
The data list/split can be created with the script `scripts/prepare_datalist.py`.

```
python scripts/prepare_datalist.py --path /path-to-Task07_Pancreas/ --output configs/dataset_0.json
```

## Training configuration
The training was performed with at least 16GB-memory GPUs.

Actual Model Input: 96 x 96 x 96

### Neural Architecture Search Configuration
The neural architecture search was performed with the following:

- AMP: True
- Optimizer: SGD
- Initial Learning Rate: 0.025
- Loss: DiceCELoss

### Optimial Architecture Training Configuration
The training was performed with the following:

- AMP: True
- Optimizer: SGD
- (Initial) Learning Rate: 0.025
- Loss: DiceCELoss

The segmentation of pancreas region is formulated as the voxel-wise 3-class classification. Each voxel is predicted as either foreground (pancreas body, tumour) or background. And the model is optimized with gradient descent method minimizing soft dice loss and cross-entropy loss between the predicted mask and ground truth segmentation.

### Input
One channel
- CT image

### Output
Three channels
- Label 2: pancreatic tumor
- Label 1: pancreas
- Label 0: everything else

## Performance
Dice score is used for evaluating the performance of the model. This model achieves a mean dice score of 0.62.

Please note that this bundle is non-deterministic because of the trilinear interpolation used in the network. Therefore, reproducing the training process may not get exactly the same performance.
Please refer to https://pytorch.org/docs/stable/notes/randomness.html#reproducibility for more details about reproducibility.

#### Training Loss
The loss over 3200 epochs (the bright curve is smoothed, and the dark one is the actual curve)

![Training loss over 3200 epochs (the bright curve is smoothed, and the dark one is the actual curve)](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_net_arch_search_segmentation_train_4-3.png)

#### Validation Dice
The mean dice score over 3200 epochs (the bright curve is smoothed, and the dark one is the actual curve)

![Validation mean dice score over 3200 epochs (the bright curve is smoothed, and the dark one is the actual curve)](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_net_arch_search_segmentation_validation_4-3.png)

### Searched Architecture Visualization
Users can install Graphviz for visualization of searched architectures (needed in [decode_plot.py](https://github.com/Project-MONAI/tutorials/blob/main/automl/DiNTS/decode_plot.py)). The edges between nodes indicate global structure, and numbers next to edges represent different operations in the cell searching space. An example of searched architecture is shown as follows:

![Example of Searched Architecture](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_net_arch_search_segmentation_searched_arch_example_1.png)

### System Memory Requirement
[The provided searched architecture and trained checkpoints](https://github.com/dongyang0122/model-zoo/blob/dev/models/pancreas_ct_dints_segmentation/large_files.yml) result from training conducted using 400GB of system RAM. Ample RAM is essential as we aim to cache all pre-processed data points in RAM to enhance the efficiency of model searching and training with [CacheDataset](https://docs.monai.io/en/stable/data.html#cachedataset). Should any errors occur during data pre-processing or caching in model search or training, kindly decrease the caching rate `cache_rate` in the configurations (e.g., [training](https://github.com/dongyang0122/model-zoo/blob/dev/models/pancreas_ct_dints_segmentation/configs/train.yaml) and [multi-GPU training](https://github.com/dongyang0122/model-zoo/blob/dev/models/pancreas_ct_dints_segmentation/configs/multi_gpu_train.yaml)) within range (0, 1) to minimize the GPU RAM requirements.

## MONAI Bundle Commands
In addition to the Pythonic APIs, a few command line interfaces (CLI) are provided to interact with the bundle. The CLI supports flexible use cases, such as overriding configs at runtime and predefining arguments in a file.

For more details usage instructions, visit the [MONAI Bundle Configuration Page](https://docs.monai.io/en/latest/config_syntax.html).

#### Execute model searching:

```
python -m scripts.search run --config_file configs/search.yaml
```

#### Execute multi-GPU model searching (recommended):

```
torchrun --nnodes=1 --nproc_per_node=8 -m scripts.search run --config_file configs/search.yaml
```

#### Execute training:

```
python -m monai.bundle run --config_file configs/train.yaml
```

#### Override the `train` config to execute multi-GPU training:

```
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run --config_file "['configs/train.yaml','configs/multi_gpu_train.yaml']"
```

#### Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run --config_file "['configs/train.yaml','configs/evaluate.yaml']"
```

#### Execute inference:

```
python -m monai.bundle run --config_file configs/inference.yaml
```

#### Export checkpoint for TorchScript:

```
python -m monai.bundle ckpt_export network_def --filepath models/model.ts --ckpt_file models/model.pt --meta_file configs/metadata.json --config_file configs/inference.yaml
```

# References

[1] He, Y., Yang, D., Roth, H., Zhao, C. and Xu, D., 2021. Dints: Differentiable neural network topology search for 3d medical image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 5841-5850).

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
