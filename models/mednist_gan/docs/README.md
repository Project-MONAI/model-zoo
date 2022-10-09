
# MedNIST GAN Hand Model

This model is a generator for creating images like the Hand category in the MedNIST dataset. It was trained as a GAN and accepts random values as inputs to produce an image output. The `train.json` file describes the training process along with the definition of the discriminator network used, and is based on the [MONAI GAN tutorials](https://github.com/Project-MONAI/tutorials/blob/main/modules/mednist_GAN_workflow_dict.ipynb).

This is a demonstration network meant to just show the training process for this sort of network with MONAI, its outputs are not particularly good and are of the same tiny size as the images in MedNIST. The training process was very short so a network with a longer training time would produce better results.

### Downloading the Dataset

Download the dataset from [here](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz) and extract the contents to a convenient location.

The MedNIST dataset was gathered from several sets from [TCIA](https://wiki.cancerimagingarchive.net/display/Public/Data+Usage+Policies+and+Restrictions),
[the RSNA Bone Age Challenge](http://rsnachallenges.cloudapp.net/competitions/4),
and [the NIH Chest X-ray dataset](https://cloud.google.com/healthcare/docs/resources/public-datasets/nih-chest).

The dataset is kindly made available by [Dr. Bradley J. Erickson M.D., Ph.D.](https://www.mayo.edu/research/labs/radiology-informatics/overview) (Department of Radiology, Mayo Clinic)
under the Creative Commons [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).


If you use the MedNIST dataset, please acknowledge the source.

### Training

Assuming the current directory is the bundle directory, and the dataset was extracted to the directory `./MedNIST`, the following command will train the network for 50 epochs:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf --bundle_root .
```

Not also the output from the training will be placed in the `models` directory but will not overwrite the `model.pt` file that may be there already. You will have to manually rename the most recent checkpoint file to `model.pt` to use the inference script mentioned below after checking the results are correct. This saved checkpoint contains a dictionary with the generator weights stored as `model` and omits the discriminator.

Another feature in the training file is the addition of sigmoid activation to the network by modifying it's structure at runtime. This is done with a line in the `training` section calling `add_module` on a layer of the network. This works best for training although the definition of the model now doesn't strictly match what it is in the `generator` section.

The generator and discriminator networks were both trained with the `Adam` optimizer with a learning rate of 0.0002 and `betas` values `[0.5, 0.999]`. These have been emperically found to be good values for the optimizer and this GAN problem.

### Inference

The included `inference.json` generates a set number of png samples from the network and saves these to the directory `./outputs`. The output directory can be changed by setting the `output_dir` value, and the number of samples changed by setting the `num_samples` value. The following command line assumes it is invoked in the bundle directory:

```
python -m monai.bundle run inferring --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf --bundle_root .
```

Note this script uses postprocessing to apply the sigmoid activation the model's outputs and to save the results to image files.


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
