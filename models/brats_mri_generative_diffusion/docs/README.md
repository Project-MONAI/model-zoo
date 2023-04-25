
# Brats MRI 3D Latent Diffusion Generative Model

This model is a generator for creating images like the Flair MRIs based on BraTS 2018 data.. It was trained as a 3d latent diffusion model and accepts Gaussian random noise as inputs to produce an image output. The `train_autoencoder.json` file describes the training process of the variational autoencoder with GAN loss, and is based on the [MONAI GAN tutorials](https://github.com/Project-MONAI/tutorials/blob/main/modules/mednist_GAN_workflow_dict.ipynb).

This is a demonstration network meant to just show the training process for this sort of network with MONAI.

### Downloading the Dataset
The training data is from the Multimodal Brain Tumor Segmentation Challenge (BraTS) 2018.

Target: 3 tumor subregions
Task: Segmentation
Modality: MRI
Size: 285 3D volumes (4 channels each)
The provided labelled data was partitioned, based on our own split, into training (200 studies), validation (42 studies) and testing (43 studies) datasets.

### Training

Assuming the current directory is the bundle directory, and the dataset was extracted to the directory `./Task01_BrainTumour`, the following command will train the autoencoder network for 1500 epochs:
```
python -m monai.bundle run --config_file configs/train_autoencoder.json
```

Or run it with multi-gpu:
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/multi_gpu_train_autoencoder.json']"
```

```
python -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/train_diffusion.json']"
```


### Inference



### Export

The generator can be exported to a Torchscript bundle with the following:

```
python -m monai.bundle ckpt_export network_def --filepath mednist_gan.ts --ckpt_file models/model.pt --meta_file configs/metadata.json --config_file configs/inference.json
```

The model can be loaded without MONAI code after this operation. For example, an image can be generated from a set of random values with:

```python
import torch
net = torch.jit.load("mednist_gan.ts")
latent = torch.rand(1, 64)
img = net(latent)  # (1,1,64,64)
```

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
