# Model Overview
A pre-trained model for volumetric (3D) segmentation of the spleen from CT images.

This model is trained using the runner-up [1] awarded pipeline of the "Medical Segmentation Decathlon Challenge 2018" using the UNet architecture [2] with 32 training images and 9 validation images.

![model workflow](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_spleen_ct_segmentation_workflow.png)

## Data
The training dataset is the Spleen Task from the Medical Segmentation Decathalon. Users can find more details on the datasets at http://medicaldecathlon.com/.

- Target: Spleen
- Modality: CT
- Size: 61 3D volumes (41 Training + 20 Testing)
- Source: Memorial Sloan Kettering Cancer Center
- Challenge: Large-ranging foreground size

## Training configuration
The segmentation of spleen region is formulated as the voxel-wise binary classification. Each voxel is predicted as either foreground (spleen) or background. And the model is optimized with gradient descent method minimizing Dice + cross entropy loss between the predicted mask and ground truth segmentation.

The training was performed with the following:

- GPU: at least 12GB of GPU memory
- Actual Model Input: 96 x 96 x 96
- AMP: True
- Optimizer: Novograd
- Learning Rate: 0.002
- Loss: DiceCELoss

### Input
One channel
- CT image

### Output
Two channels
- Label 1: spleen
- Label 0: everything else

## Performance
Dice score is used for evaluating the performance of the model. This model achieves a mean dice score of 0.959.

#### Training Loss
![A graph showing the training loss over 1260 epochs (10080 iterations).](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_spleen_ct_segmentation_train_3.png)

#### Validation Dice
![A graph showing the validation mean Dice over 1260 epochs.](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_spleen_ct_segmentation_val_3.png)

#### TensorRT speedup
The `spleen_ct_segmentation` bundle supports the TensorRT acceleration. The table below shows the speedup ratios benchmarked on an A100 80G GPU. The `model computation` means the speedup ratio of model's inference with a random input without preprocessing and postprocessing. The `model computation(onnx)` basically means the same thing as the `model computation`, except that the model is converted through the onnx-torchscript way. We add this line in the table since it has a better performance than the model converted through Torch-TensorRT. The `end2end` means run the bundle end to end with the TensorRT based model converted through Torch-TensorRT. The `torch_fp32` and `torch_amp` is for the pytorch model with or without `amp` mode. The `trt_fp32` and `trt_fp16` is for the TensorRT based model converted in corresponding precision. The `speedup amp`, `speedup fp32` and `speedup fp16` is the speedup ratio of corresponding models versus the pytorch float32 model, while the `amp vs fp16` is between the pytorch amp model and the TensorRT float16 based model.

| method | torch_fp32(ms) | torch_amp(ms) | trt_fp32(ms) | trt_fp16(ms) | speedup amp | speedup fp32 | speedup fp16 | amp vs fp16|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| model computation | 6.48 | 4.48 | 6.40 | 6.30 | 1.45 | 1.01 | 1.03 | 0.71 |
| model computation(onnx) | 6.46 | 4.48 | 2.52 | 1.96 | 1.44 | 2.56 | 3.30 | 2.29 |
| end2end | 3900.73 | 3823.89 | 3887.37 | 3883.01 |	1.02 | 1.00 | 1.00 | 0.98 |

This result is benchmarked under:
 - TensorRT: 8.5.3+cuda11.8
 - Torch-TensorRT Version: 1.4.0
 - CPU Architecture: x86-64
 - OS: ubuntu 20.04
 - Python version:3.8.10
 - CUDA version: 11.8
 - GPU models and configuration: A100 80G

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

#### Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run --config_file "['configs/train.json','configs/evaluate.json']"
```

#### Override the `train` config and `evaluate` config to execute multi-GPU evaluation:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run --config_file "['configs/train.json','configs/evaluate.json','configs/multi_gpu_evaluate.json']"
```

#### Execute inference:

```
python -m monai.bundle run --config_file configs/inference.json
```

#### Export checkpoint to TensorRT based models with fp32 or fp16 precision:

```
python -m monai.bundle trt_export --net_id network_def --filepath models/model_trt.ts --ckpt_file models/model.pt --meta_file configs/metadata.json --config_file configs/inference.json --precision <fp32/fp16> --dynamic_batchsize "[1, 4, 8]"
```

#### Execute inference with the TensorRT model:

```
python -m monai.bundle run --config_file "['configs/inference.json', 'configs/inference_trt.json']"
```

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
