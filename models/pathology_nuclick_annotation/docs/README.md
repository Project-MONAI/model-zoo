# Description
A pre-trained model for segmenting nuclei cells with user clicks/interactions.

![nuclick](https://github.com/mostafajahanifar/nuclick_torch/raw/master/docs/11.gif)
![nuclick](https://github.com/mostafajahanifar/nuclick_torch/raw/master/docs/33.gif)
![nuclick](https://github.com/mostafajahanifar/nuclick_torch/raw/master/docs/22.gif)

# Model Overview
This model is trained using [BasicUNet](https://docs.monai.io/en/latest/networks.html#basicunet) over [ConSeP](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet) dataset.

## Data
The training dataset is from https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet
```commandline
wget https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep_dataset.zip
unzip -q consep_dataset.zip
```
![](images/dataset.jpeg)<br/>

## Training configuration
The training was performed with the following:

- GPU: at least 12GB of GPU memory
- Actual Model Input: 5 x 128 x 128
- AMP: True
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss: DiceLoss


### Preprocessing
After [downloading this dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep_dataset.zip),
python script `data_process.py` from `scripts` folder can be used to preprocess and generate the final dataset for training.

```
python scripts/data_process.py --input /path/to/data/CoNSeP --output /path/to/data/CoNSePNuclei
```

After generating the output files, please modify the `dataset_dir` parameter specified in `configs/train.json` and `configs/inference.json` to reflect the output folder which contains new dataset.json.

Class values in dataset are

 - 1 = other
 - 2 = inflammatory
 - 3 = healthy epithelial
 - 4 = dysplastic/malignant epithelial
 - 5 = fibroblast
 - 6 = muscle
 - 7 = endothelial

As part of pre-processing, the following steps are executed.

 - Crop and Extract each nuclei Image + Label (128x128) based on the centroid given in the dataset.
 - Combine classes 3 & 4 into the epithelial class and 5,6 & 7 into the spindle-shaped class.
 - Update the label index for the target nuclie based on the class value
 - Other cells which are part of the patch are modified to have label idex = 255

Example dataset.json
```json
{
  "training": [
    {
      "image": "/workspace/data/CoNSePNuclei/Train/Images/train_1_3_0001.png",
      "label": "/workspace/data/CoNSePNuclei/Train/Labels/train_1_3_0001.png",
      "nuclei_id": 1,
      "mask_value": 3,
      "centroid": [
        64,
        64
      ]
    }
  ],
  "validation": [
    {
      "image": "/workspace/data/CoNSePNuclei/Test/Images/test_1_3_0001.png",
      "label": "/workspace/data/CoNSePNuclei/Test/Labels/test_1_3_0001.png",
      "nuclei_id": 1,
      "mask_value": 3,
      "centroid": [
        64,
        64
      ]
    }
  ]
}
```


## Input and output formats
### Input: 5 channels
- 3 RGB channels
- +ve signal channel (this nuclei)
- -ve signal channel (other nuclei)

### Output: 2 channels
 - 0 = Background
 - 1 = Nuclei

![](images/train_in_out.jpeg)

## Scores
This model achieves the following Dice score on the validation data provided as part of the dataset:

- Train Dice score = 0.89
- Validation Dice score = 0.85


## Training Performance
A graph showing the training Loss and Dice over 50 epochs.

![](images/train_loss.jpeg) <br>
![](images/train_dice.jpeg) <br>

## Validation Performance
A graph showing the validation mean Dice over 50 epochs.

![](images/val_dice.jpeg) <br>


## commands example
Execute training:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf
```

Override the `train` config to execute multi-GPU training:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run training --meta_file configs/metadata.json --config_file "['configs/train.json','configs/multi_gpu_train.json']" --logging_file configs/logging.conf
```

Please note that the distributed training related options depend on the actual running environment, thus you may need to remove `--standalone`, modify `--nnodes` or do some other necessary changes according to the machine you used.
Please refer to [pytorch's official tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for more details.

Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.json','configs/evaluate.json']" --logging_file configs/logging.conf
```

Override the `train` config and `evaluate` config to execute multi-GPU evaluation:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.json','configs/evaluate.json','configs/multi_gpu_evaluate.json']" --logging_file configs/logging.conf
```

Execute inference:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf
```

# Disclaimer
This is an example, not to be used for diagnostic purposes.

# References
[1] Koohbanani, Navid Alemi, et al. "NuClick: a deep learning framework for interactive segmentation of microscopic images." Medical Image Analysis 65 (2020): 101771. https://arxiv.org/abs/2005.14511.

[2] S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, Sept. 2019. [[doi](https://doi.org/10.1016/j.media.2019.101563)]

[3] NuClick [PyTorch](https://github.com/mostafajahanifar/nuclick_torch) Implementation

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
