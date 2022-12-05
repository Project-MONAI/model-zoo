# Model Overview

A pre-trained model for simultaneous segmentation and classification of nuclei within multi-tissue histology images based on CoNSeP data. The details of the model can be found in [1].

## Workflow

The model is trained to simultaneous segment and classify nuclei. Training is done via a two-stage approach. First initialized the model with pre-trained weights on the [ImageNet dataset](https://ieeexplore.ieee.org/document/5206848), trained only the decoders for the first 50 epochs, and then fine-tuned all layers for another 50 epochs.

- Each user is responsible for checking the content of models/datasets and the applicable licenses and determining if suitable for the intended use.The license for the pre-trained model used in examples is different than MONAI license. Please check the source where these weights are obtained from: <https://github.com/vqdang/hover_net#data-format>

`pretrained_model` is "https://drive.google.com/u/1/uc?id=1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5&export=download" which will be used in bash code below.

![Model workflow](https://ars.els-cdn.com/content/image/1-s2.0-S1361841519301045-fx1_lrg.jpg)

## Data

The training data is from <https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/>.

- Target: segment instance-level nuclei and classify the nuclei type
- Task: Segmentation and classification
- Modality: RGB images
- Size: 41 image tiles (2009 patches)

The provided labelled data was partitioned, based on the original split, into training (27 tiles) and testing (14 tiles) datasets.

After download the datasets, please run `scripts/prepare_patches.py` to prepare patches from tiles. Prepared patches are saved in `your-concep-dataset-path`/Prepared. The implementation is referring to <https://github.com/vqdang/hover_net/blob/master/extract_patches.py>. The command is like:

```
python scripts/prepare_patches.py -root your-concep-dataset-path
```

## Training configuration

This model utilized a two-stage approach. The training was performed with the following:

- GPU: At least 24GB of GPU memory.
- Actual Model Input: 270 x 270
- AMP: True
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss: HoVerNetLoss

## Input

Input: RGB images

1. Randomly performing rotate, scale, shear, translate, etc
2. Cropping images to (270, 270)
3. Randomly spatial flipping
4. Randomly choosing one transform to smooth images
5. Randomly adjusting brightness, saturation, contrast and hue
6. Scaleing intensity to [0, 1]
7. Generating horizontal and vertical distance maps from label instance map

## Output

Output: a dictionary with the following keys:

1. nucleus_prediction: predict whether or not a pixel belongs to the nuclei or background
2. horizontal_vertical: predict the horizontal and vertical distances of nuclear pixels to their centres of mass
3. type_prediction: predict the type of nucleus for each pixel

## Model Performance

The achieved metrics on the validation data are:

- Binary Dice: 0.82762
- PQ: 0.48976
- F1d: 0.73592

#### Training Loss and Dice

stage1:
![A graph showing the training loss and the mean dice over 50 epochs in stage1](https://drive.google.com/file/d/1sc0ZqKWmphZbhHxKu-UredYriFtUFlbk/view?usp=sharing)

stage2:
![A graph showing the training loss and the mean dice over 50 epochs in stage2](https://drive.google.com/file/d/1zdflo7lr6E4tX6YzoTUQItqlJwIbHBEg/view?usp=sharing)

#### Validation Dice

stage1:

![A graph showing the validation mean dice over 50 epochs in stage1](https://drive.google.com/file/d/1skc_BqdqHht7r1zV288lq9XKUiH83hHe/view?usp=sharing)

stage2:

![A graph showing the validation mean dice over 50 epochs in stage2](https://drive.google.com/file/d/1hjO6vOnU5zXwrNqu3gpKOdcrABIcy1Ua/view?usp=sharing)

## commands example

Execute training:

- Run first stage

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf --network_def#pretrained_url `pretrained_model` --stage 0
```

- Run second stage

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf --network_def#freeze_encoder false --network_def#pretrained_url None --stage 1
```

Override the `train` config to execute multi-GPU training:

- Run first stage

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run training --meta_file configs/metadata.json --config_file "['configs/train.json','configs/multi_gpu_train.json']" --logging_file configs/logging.conf --train#dataloader#batch_size 8 --network_def#freeze_encoder true --network_def#pretrained_url `pretrained_model` --stage 0
```

- Run second stage

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run training --meta_file configs/metadata.json --config_file "['configs/train.json','configs/multi_gpu_train.json']" --logging_file configs/logging.conf --train#dataloader#batch_size 4 --network_def#freeze_encoder false --network_def#pretrained_url None --stage 1
```

Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.json','configs/evaluate.json']" --logging_file configs/logging.conf
```

### Execute inference

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf
```

# Disclaimer

This is an example, not to be used for diagnostic purposes.

# References

[1] Simon Graham. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, 2019. <https://arxiv.org/abs/1812.06499>
