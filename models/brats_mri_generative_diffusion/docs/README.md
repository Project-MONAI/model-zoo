# Model Overview
A pre-trained model for volumetric (3D) Brats MRI 3D Latent Diffusion Generative Model.

This model is trained on BraTS 2016 and 2017 data from [Medical Decathlon](http://medicaldecathlon.com/), using the Latent diffusion model [1].

![model workflow](https://developer.download.nvidia.com/assets/Clara/Images/monai_brain_image_gen_ldm3d_network.png)

This model is a generator for creating images like the Flair MRIs based on BraTS 2016 and 2017 data. It was trained as a 3d latent diffusion model and accepts Gaussian random noise as inputs to produce an image output. The `train_autoencoder.json` file describes the training process of the variational autoencoder with GAN loss. The `train_diffusion.json` file describes the training process of the 3D latent diffusion model.

**This is a demonstration network meant to just show the training process for this sort of network with MONAI. To achieve better performance, users need to use larger dataset like [Brats 2021](https://www.synapse.org/#!Synapse:syn25829067/wiki/610865) and have GPU with memory larger than 32G to enable larger networks and attention layers.**

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
The training data is Brats 2016 and 2017 from [Medical Decathlon](http://medicaldecathlon.com/).

Target: image generatiion
Task: Synthesis
Modality: MRI
Size: 388 3D volumes (1 channel used)

The dataset can be downloaded automatically at the beggining of training.

## Training configuration
If user has GPU memory smaller than 32G, then please decrease the `"train_batch_size"` in `configs/train_autoencoder.json` and `configs/train_diffusion.json`.

### Training configuration of autoencoder
The training of autoencoder was performed with the following:

- GPU: at least 32GB GPU memory
- Actual Model Input: 112 x 128 x 80
- AMP: False
- Optimizer: Adam
- Learning Rate: 1e-5
- Loss: L1 loss, perceptual loss, KL divergence loss, adversianl loss, GAN BCE loss

#### Input
1 channel 3D MRI Flair patches

#### Output
- 1 channel 3D MRI reconstructed patches
- 8 channel mean of latent features
- 8 channel standard deviation of latent features

### Training configuration of diffusion model
The training of latent diffusion model was performed with the following:

- GPU: at least 32GB GPU memory
- Actual Model Input: 36 x 44 x 28
- AMP: False
- Optimizer: Adam
- Learning Rate: 1e-5
- Loss: MSE loss

#### Training Input
- 8 channel noisy latent features
- an int that indicates the time step

#### Training Output
8 channel predicted added noise

#### Inference Input
8 channel noise

#### Inference Output
8 channel denoised latent features

### Memory Consumption Warning

If you face memory issues with data loading, you can lower the caching rate `cache_rate` in the configurations within range [0, 1] to minimize the System RAM requirements.

## Performance

#### Training Loss
![A graph showing the autoencoder training curve](https://developer.download.nvidia.com/assets/Clara/Images/monai_brain_image_gen_ldm3d_train_autoencoder_loss.png)

![A graph showing the latent diffusion training curve](https://developer.download.nvidia.com/assets/Clara/Images/monai_brain_image_gen_ldm3d_train_diffusion_loss.png)

#### Example synthetic image
![Example synthetic image](https://developer.download.nvidia.com/assets/Clara/Images/monai_brain_image_gen_ldm3d_example_generation.png)

## MONAI Bundle Commands

In addition to the Pythonic APIs, a few command line interfaces (CLI) are provided to interact with the bundle. The CLI supports flexible use cases, such as overriding configs at runtime and predefining arguments in a file.

For more details usage instructions, visit the [MONAI Bundle Configuration Page](https://docs.monai.io/en/latest/config_syntax.html).

#### Execute training:

- Train autoencoder
```
python -m monai.bundle run --config_file configs/train_autoencoder.json
```

Please specify "download_brats" into `True` if the dataset is not downloaded.

- Train latent diffusion model
```
python -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/train_diffusion.json']"
```

It will print out the scale factor of the latent feature space. If your autoencoder is well trained, this value should be close to 1.0.

#### Override the `train` config to execute multi-GPU autoencoder training:

Run with multi-GPU requires the learning rate to be scaled up according to the number of GPUs.

- Train autoencoder

```
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/multi_gpu_train_autoencoder.json']" --lr 8e-5
```

- Train latent difussion model

```
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/train_diffusion.json','configs/multi_gpu_train_autoencoder.json','configs/multi_gpu_train_diffusion.json']"  --lr 8e-5
```

#### Execute inference:

The following code generates a synthetic image from a random sampled noise.
```
python -m monai.bundle run --config_file configs/inference.json
```

#### Export checkpoint to TorchScript file:

The autoencoder can be exported into a TorchScript file.

```
python -m monai.bundle ckpt_export autoencoder_def --filepath models/model_autoencoder.ts --ckpt_file models/model_autoencoder.pt --meta_file configs/metadata.json --config_file "['configs/train_autoencoder.json','configs/train_diffusion.json','configs/inference.json']"
```

# References
[1] Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022. https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf

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
