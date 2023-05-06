# Model Overview
A pre-trained model for volumetric (3D) Brats MRI 3D Latent Diffusion Generative Model.

This model is trained on BraTS 2018 data (https://www.med.upenn.edu/sbia/brats2018.html), using the Latent diffusion model (Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022. https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf).

![model workflow](https://miro.medium.com/v2/resize:fit:1256/1*3Jjlrb-TB8hPpIexoCKpZQ.png)

This model is a generator for creating images like the Flair MRIs based on BraTS 2018 data. It was trained as a 3d latent diffusion model and accepts Gaussian random noise as inputs to produce an image output. The `train_autoencoder.json` file describes the training process of the variational autoencoder with GAN loss. The `train_diffusion.json` file describes the training process of the 3D latent diffusion model.

This is a demonstration network meant to just show the training process for this sort of network with MONAI. To achieve better performance, users need to have GPU with memory larger than 32G to enable larger networks and attention layers.

## Install the dependency of MONAI generative models
[MONAI generative models](https://github.com/Project-MONAI/GenerativeModels) can be installed by
```
pip install lpips
pip install git+https://github.com/Project-MONAI/GenerativeModels.git@0.2.0
```

Or we can install it from source
```
pip install lpips
git clone https://github.com/Project-MONAI/GenerativeModels.git
cd GenerativeModels/
python setup.py install
cd ..
```

## Downloading the Dataset
The training data is from the [Multimodal Brain Tumor Segmentation Challenge (BraTS) 2018](https://www.med.upenn.edu/sbia/brats2018.html).

Target: image generatiion
Task: Synthesis
Modality: MRI
Size: 388 3D volumes (1 channel used)

The dataset can be downloaded automatically at the beggining of training.

## Training configuration
If user has GPU memory smaller than 32G, then please decrease the `"train_batch_size_img"` and `"train_batch_size_slice"` in `configs/train_autoencoder.json` and `configs/train_diffusion.json`.
`"train_batch_size_img"` indicates how many 3D images in each batch. `"train_batch_size_slice"` indicates how many 2D slices it extracts from each 3D image. The product of these two hyper-parameters is the actual training batchsize.

### Training configuration of autoencoder
The training of autoencoder was performed with the following:

- GPU: at least 32GB GPU memory
- Actual Model Input: 240x240
- AMP: False
- Optimizer: Adam
- Learning Rate: 1e-5
- Loss: L1 loss, perceptual loss, KL divergence loss, adversianl loss, GAN BCE loss

#### Input
1 channel 2D MRI Flair axial slices

#### Output
- 1 channel 2D MRI reconstructed axial slices
- 1 channel mean of latent features
- 1 channel standard deviation of latent features

### Training configuration of difusion model
The training of latent diffusion model was performed with the following:

- GPU: at least 32GB GPU memory
- Actual Model Input: 64x64
- AMP: False
- Optimizer: Adam
- Learning Rate: 1e-5
- Loss: MSE loss

#### Training Input
1 channel noisy latent features

#### Training Output
1 channel predicted added noise

#### Inference Input
1 channel noise

#### Inference Output
1 channel denoised latent features


## MONAI Bundle Commands

Assuming the current directory is the bundle directory, the following command will train the autoencoder network for 1500 epochs.
If the Brats dataset is not downloaded, run the following command, the data will be downloaded and extracted to `./Task01_BrainTumour`.
```
python -m monai.bundle run --config_file configs/train_autoencoder.json --dataset_dir ./ --download_brats True
```
If it is already downloaded, check if `"dataset_dir"` in `configs/train_autoencoder.json`has folder `Task01_BrainTumour`. If so, run:
```
python -m monai.bundle run --config_file configs/train_autoencoder.json
```

Or run it with multi-gpu, which requires the learning rate to be scaled up according to the number of GPUs.
```
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/multi_gpu_train_autoencoder.json']" --lr 1e-4
```
It take 21 hours when training with 8 GPU, each using 32G memory.

After the autoencoder is trained, run the following command to train the latent diffusion model.
```
python -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/train_diffusion.json']"
```
It will print out the scale factor of the latent feature space. If your autoencoder is well trained, this value should be close to 1.0.

Or run it with multi-gpu, which requires the learning rate to be scaled up according to the number of GPUs.
```
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/train_diffusion.json','configs/multi_gpu_train_autoencoder.json','configs/multi_gpu_train_diffusion.json']"  --lr 1e-4
```

It take 13 hours when training with 8 GPU, each using 32G memory.

<p align="center">
  <img src="https://developer.download.nvidia.com/assets/Clara/Images/monai_brain_image_gen_ldm2d_train_autoencoder_loss.png" alt="autoencoder training curve" width="45%" >
&nbsp; &nbsp; &nbsp; &nbsp;
  <img src="https://developer.download.nvidia.com/assets/Clara/Images/monai_brain_image_gen_ldm2d_train_diffusion_loss.png" alt="latent diffusion training curve" width="45%" >
</p>

### Inference
The following code generates a synthetic image from a random sampled noise.
```
python -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/train_diffusion.json','configs/inference.json']"
```
The generated image will be saved to `./output/0`

An example output is shown below. Note that this is a demonstration network meant to just show the training process for this sort of network with MONAI. To achieve better performance, users need to have GPU with memory larger than 32G to enable larger networks and attention layers.

![Example synthetic image](placeholder)


### Export

The autoencoder can be exported to a Torchscript bundle with the following:
```
python -m monai.bundle ckpt_export autoencoder_def --filepath models/model_autoencoder.ts --ckpt_file models/model_autoencoder.pt --meta_file configs/metadata.json --config_file "['configs/train_autoencoder.json','configs/train_diffusion.json','configs/inference.json']"
```

The models can be loaded after this operation:
```python
import torch
autoencoder = torch.jit.load("models/model_autoencoder.ts")
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