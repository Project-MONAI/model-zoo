# Prostate MRI zonal segmentation

### **Authors**

Lisa C. Adams, Keno K. Bressem

### **Tags**

Segmentation, MR, Prostate

## **Model Description**
This model was trained with the UNet architecture [1] and is used for 3D volumetric segmentation of the anatomical prostate zones on T2w MRI images. The segmentation of the anatomical regions is formulated as a voxel-wise classification. Each voxel is classified as either central gland (1), peripheral zone (2), or background (0). The model is optimized using a gradient descent method that minimizes the focal soft-dice loss between the predicted mask and the actual segmentation.

## **Data**
The model was trained in the prostate158 training data, which is available at https://doi.org/10.5281/zenodo.6481141. Only T2w images were used for this task.


### **Preprocessing**
MRI images in the prostate158 dataset were preprocessed, including center cropping and resampling. When applying the model to new data, this preprocessing should be repeated.

#### **Center cropping**
T2w images were acquired with a voxel spacing of 0.47 x 0.47 x 3 mm and an axial FOV size of 180 x 180 mm. However, the prostate rarely exceeds an axial diameter of 100 mm, and for zonal segmentation, the tissue surrounding the prostate is not of interest and only increases the image size and thus the computational cost. Center-cropping can reduce the image size without sacrificing information.

The script `center_crop.py` allows to reproduce center-cropping as performed in the prostate158 paper.

```bash
python scripts/center_crop.py --file_name path/to/t2_image --out_name cropped_t2
```

#### **Resampling**
DWI and ADC sequences in prostate158 were resampled to the orientation and voxel spacing of the T2w sequence. As the zonal segmentation uses T2w images, no additional resampling is nessecary. However, the training script will perform additonal resampling automatically.


## **Performance**
The model achives the following performance on the prostate158 test dataset:

<table border=1 frame=void rules=rows>
    <thead>
        <tr>
            <td></td>
            <td colspan = 3><b><center>Rater 1</center></b></td>
            <td>&emsp;</td>
            <td colspan = 3><b><center>Rater 2</center></b></td>
        </tr>
        <tr>
            <th>Metric</th>
            <th>Transitional Zone</th>
            <th>Peripheral Zone</th>
            <th>&emsp;</th>
            <th>Transitional Zone</th>
            <th>Peripheral Zone</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href='https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient'>Dice Coefficient </a></td>
            <td> 0.877</td>
            <td> 0.754</td>
            <td>&emsp;</td>
            <td> 0.875</td>
            <td> 0.730</td>
        </tr>
        <tr>
            <td><a href='https://en.wikipedia.org/wiki/Hausdorff_distance'>Hausdorff Distance </a></td>
            <td> 18.3</td>
            <td> 22.8</td>
            <td>&emsp;</td>
            <td> 17.5</td>
            <td> 33.2</td>
        </tr>
        <tr>
            <td><a href='https://github.com/deepmind/surface-distance'>Surface Distance </a></td>
            <td> 2.19</td>
            <td> 1.95</td>
            <td>&emsp;</td>
            <td> 2.59</td>
            <td> 1.88</td>
        </tr>
    </tbody>
</table>

For more details, please see the original [publication](https://doi.org/10.1016/j.compbiomed.2022.105817) or official [GitHub repository](https://github.com/kbressem/prostate158)


## **System Configuration**
The model was trained for 100 epochs on a workstaion with a single Nvidia RTX 3080 GPU. This takes approximatly 8 hours.

## **Limitations** (Optional)

This training and inference pipeline was developed for research purposes only. This research use only software that has not been cleared or approved by FDA or any regulatory agency. The model is for research/developmental purposes only and cannot be used directly for clinical procedures.

## **Citation Info** (Optional)

```
@article{ADAMS2022105817,
title = {Prostate158 - An expert-annotated 3T MRI dataset and algorithm for prostate cancer detection},
journal = {Computers in Biology and Medicine},
volume = {148},
pages = {105817},
year = {2022},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2022.105817},
url = {https://www.sciencedirect.com/science/article/pii/S0010482522005789},
author = {Lisa C. Adams and Marcus R. Makowski and Günther Engel and Maximilian Rattunde and Felix Busch and Patrick Asbach and Stefan M. Niehues and Shankeeth Vinayahalingam and Bram {van Ginneken} and Geert Litjens and Keno K. Bressem},
keywords = {Prostate cancer, Deep learning, Machine learning, Artificial intelligence, Magnetic resonance imaging, Biparametric prostate MRI}
}
```

## **References**

[1] Sakinis, Tomas, et al. "Interactive segmentation of medical images through fully convolutional neural networks." arXiv preprint arXiv:1903.08205 (2019).

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
