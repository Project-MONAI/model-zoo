
# Brats MRI 3D Latent Diffusion Generative Model

This model is a generator for creating images like the Flair MRIs based on BraTS 2018 data.. It was trained as a 3d latent diffusion model and accepts Gaussian random noise as inputs to produce an image output. The `train_autoencoder.json` file describes the training process of the variational autoencoder with GAN loss. The `train_diffusion.json` file describes the training process of the 3D latent diffusion model.

This is a demonstration network meant to just show the training process for this sort of network with MONAI.

### Install the dependency of MONAI generative models
[MONAI generative models](https://github.com/Project-MONAI/GenerativeModels) can be installed by
```
git clone https://github.com/Project-MONAI/GenerativeModels.git
cd GenerativeModels/
python setup.py install
cd ..
```
We also need
```
pip install lpips
```

### Downloading the Dataset
The training data is from the [Multimodal Brain Tumor Segmentation Challenge (BraTS) 2018](https://www.med.upenn.edu/sbia/brats2018.html).

Target: image generatiion
Task: Synthesis
Modality: MRI
Size: 285 3D volumes (1 channel used)

The dataset can be downloaded automatically at the beggining of training.

### Training

Assuming the current directory is the bundle directory, the following command will train the autoencoder network for 1500 epochs.
If the Brats dataset is not downloaded, run the following command, the data will be downloaded and extracted to `./Task01_BrainTumour`.
```
python -m monai.bundle run --config_file configs/train_autoencoder.json --dataset_dir ./ --download_brats True
```
If it is already downloaded, check if `"dataset_dir"` in `configs/train_autoencoder.json`has folder `Task01_BrainTumour`. If so, run:
```
python -m monai.bundle run --config_file configs/train_autoencoder.json
```

Or run it with multi-gpu:
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/multi_gpu_train_autoencoder.json']"
```
It take 9 hours when training with 9 32G GPU.

After the autoencoder is trained, run the following command to train the latent diffusion model.
```
python -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/train_diffusion.json']"
```
It will print out the scale factor of the latent feature space. If your autoencoder is well trained, this value should be close to 1.0.

Or run it with multi-gpu:
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/train_diffusion.json','configs/multi_gpu_train_diffusion.json']"
```


### Inference
```
python -m monai.bundle save_nii --config_file "['configs/train_autoencoder.json','configs/train_diffusion.json','configs/inference.json']"
```



### Export

The generator can be exported to a Torchscript bundle with the following:

```
python -m monai.bundle ckpt_export autoencoder_def --filepath autoencoder.ts --ckpt_file models/model_autoencoder.pt --meta_file configs/metadata.json --config_file configs/inference.json
```

The model can be loaded without MONAI code after this operation. For example, an image can be generated from a set of random values with:

```python
import torch
net = torch.jit.load("autoencoder.ts")
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
