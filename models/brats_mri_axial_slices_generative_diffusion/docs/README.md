# Model Overview
A pre-trained model for 2D Latent Diffusion Generative Model on axial slices of BraTS MRI.

This model is trained on BraTS 2016 and 2017 data from [Medical Decathlon](http://medicaldecathlon.com/), using the Latent diffusion model (Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022. https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf).

![model workflow](https://developer.download.nvidia.com/assets/Clara/Images/monai_brain_image_gen_ldm3d_network.png)

This model is a generator for creating images like the Flair MRIs based on BraTS 2016 and 2017 data. It was trained as a 2d latent diffusion model and accepts Gaussian random noise as inputs to produce an image output. The `train_autoencoder.json` file describes the training process of the variational autoencoder with GAN loss. The `train_diffusion.json` file describes the training process of the 2D latent diffusion model.

#### Example synthetic image
An example result from inference is shown below:
![Example synthetic image](https://developer.download.nvidia.com/assets/Clara/Images/monai_brain_image_gen_ldm2d_example_generation.png)

**This is a demonstration network meant to just show the training process for this sort of network with MONAI. To achieve better performance, users need to use larger dataset like [BraTS 2021](https://www.synapse.org/#!Synapse:syn25829067/wiki/610865).**

## MONAI Generative Model Dependencies
[MONAI generative models](https://github.com/Project-MONAI/GenerativeModels) can be installed by
```
pip install lpips==0.4.1
pip install git+https://github.com/Project-MONAI/GenerativeModels.git@0.2.0
```

Or we can install it from source
```
pip install lpips==0.4.1
git clone https://github.com/Project-MONAI/GenerativeModels.git
cd GenerativeModels/
python setup.py install
cd ..
```

## Data
The training data is BraTS 2016 and 2017 from [Medical Decathlon](http://medicaldecathlon.com/).

- Target: Image Generation
- Task: Synthesis
- Modality: MRI
- Size: 388 3D MRI volumes (1 channel used)
- Training data size: 38800 2D MRI axial slices (1 channel used)

The dataset can be downloaded automatically at the beginning of training.

## Training Configuration
If you have a GPU with less than 32G of memory, you may need to decrease the batch size when training. To do so, modify the `"train_batch_size_img"` and `"train_batch_size_slice"` parameters in the `configs/train_autoencoder.json` and `configs/train_diffusion.json` configuration files.
- `"train_batch_size_img"` is number of 3D volumes loaded in each batch.
- `"train_batch_size_slice"` is the number of 2D axial slices extracted from each image. The actual batch size is the product of them.

In this bundle, the autoencoder uses perceptual loss, which is based on a model with pre-trained weights from some internal data. This model is frozen and will not be trained in the bundle.
The path of the model is specified in `perceptual_loss_model_weights_path` parameter in the [configs/train_autoencoder.json](configs/train_autoencoder.json). The [MONAI Generative Model repo](https://github.com/Project-MONAI/GenerativeModels/blob/fd04ec6f98a1aec7b6886dff1cfb4d0fa72fe4fe/generative/losses/perceptual.py#L64-L69) and [torchvison](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#ResNet50_Weights) also provide pre-trained weights but may be for non-commercial use only. Each user is responsible for checking the data source of the pre-trained models, the applicable licenses, and determining if suitable for the intended use.

### Training Configuration of Autoencoder
The autoencoder was trained using the following configuration:

- GPU: at least 32GB GPU memory
- Actual Model Input: 240 x 240
- AMP: False
- Optimizer: Adam
- Learning Rate: 2e-5
- Loss: L1 loss, perceptual loss, KL divergence loss, adversarial  loss, GAN BCE loss

#### Input
1 channel 2D MRI Flair axial patches

#### Output
- 1 channel 2D MRI reconstructed patches
- 1 channel mean of latent features
- 1 channel standard deviation of latent features

### Training Configuration of Diffusion Model
The latent diffusion model was trained using the following configuration:

- GPU: at least 32GB GPU memory
- Actual Model Input: 64 x 64
- AMP: False
- Optimizer: Adam
- Learning Rate: 1e-5
- Loss: MSE loss

#### Training Input
- 1 channel noisy latent features
- an int that indicates the time step

#### Training Output
1 channel predicted added noise

#### Inference Input
1 channel noise

#### Inference Output
1 channel denoised latent features

## Performance

Below is an example of the loss values for both the Autoencoder and the Latent Diffusion Model during training:

<p align="center">
  <img src="https://developer.download.nvidia.com/assets/Clara/Images/monai_brain_image_gen_ldm2d_train_autoencoder_loss_v2.png" alt="autoencoder training curve" width="45%" >
&nbsp; &nbsp; &nbsp; &nbsp;
  <img src="https://developer.download.nvidia.com/assets/Clara/Images/monai_brain_image_gen_ldm2d_train_diffusion_loss_v2.png" alt="latent diffusion training curve" width="45%" >
</p>

Keep in mind that actual performance will depend on a variety of factors, such as the size of the dataset, the quality of the images, and the training parameters used.


## MONAI Bundle Commands
In addition to the Pythonic APIs, a few command line interfaces (CLI) are provided to interact with the bundle. The CLI supports flexible use cases, such as overriding configs at runtime and predefining arguments in a file.

For more details usage instructions, visit the [MONAI Bundle Configuration Page](https://docs.monai.io/en/latest/config_syntax.html).

#### Execute Autoencoder Training (w/data download)

Assuming the current directory is the bundle directory, the following command will train the autoencoder network for 1500 epochs using the BraTS dataset. If the dataset is not downloaded, it will be automatically downloaded and extracted to `./Task01_BrainTumour`.

```
python -m monai.bundle run --config_file configs/train_autoencoder.json --dataset_dir ./ --download_brats True
```

#### Execute Autoencoder Training
If the dataset is already downloaded, make sure that `"dataset_dir"` in `configs/train_autoencoder.json` has the correct path to the dataset `Task01_BrainTumour`. Then, run:

```
python -m monai.bundle run --config_file configs/train_autoencoder.json
```

#### Override the `train` config to execute multi-GPU training for Autoencoder:
To train with multiple GPUs, use the following command, which requires scaling up the learning rate according to the number of GPUs. Keep in mind that this command will take approximately 9 hours to complete when using 8 GPUs, each with 32G of memory.

```
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/multi_gpu_train_autoencoder.json']" --lr 4e-4
```

#### Execute Latent Diffusion Model Training
After training the autoencoder, run the following command to train the latent diffusion model. This command will print out the scale factor of the latent feature space. If your autoencoder is well trained, this value should be close to 1.0.

```
python -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/train_diffusion.json']"
```

#### Override the `train` config to execute multi-GPU training for Latent Diffusion Model:
To train with multiple GPUs, use the following command, which requires scaling up the learning rate according to the number of GPUs. Keep in mind that this command will take approximately 5 hours to complete when using 8 GPUs, each with 32G of memory.
```
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m monai.bundle run --config_file "['configs/train_autoencoder.json','configs/train_diffusion.json','configs/multi_gpu_train_autoencoder.json','configs/multi_gpu_train_diffusion.json']"  --lr 2e-4
```


#### Execute inference:
The following code generates a synthetic image from a random sampled noise.
```
python -m monai.bundle run --config_file configs/inference.json
```
The generated image will be saved to `./output`



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
