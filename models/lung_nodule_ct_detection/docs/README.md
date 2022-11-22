# Model Overview
A pre-trained model for volumetric (3D) detection of the lung nodule from CT image.

This model is trained on LUNA16 dataset (https://luna16.grand-challenge.org/Home/), using the RetinaNet (Lin, Tsung-Yi, et al. "Focal loss for dense object detection." ICCV 2017. https://arxiv.org/abs/1708.02002).

<p align="center">
  <img src="https://developer.download.nvidia.com/assets/Clara/Images/monai_retinanet_detection_workflow.png" alt="detection scheme")
</p>

## 1. Data
### 1.1 Data description
The dataset we are experimenting in this example is LUNA16 (https://luna16.grand-challenge.org/Home/), which is based on [LIDC-IDRI database](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) [3,4,5].

LUNA16 is a public dataset of CT lung nodule detection. Using raw CT scans, the goal is to identify locations of possible nodules, and to assign a probability for being a nodule to each location.

Disclaimer: We are not the host of the data. Please make sure to read the requirements and usage policies of the data and give credit to the authors of the dataset! We acknowledge the National Cancer Institute and the Foundation for the National Institutes of Health, and their critical role in the creation of the free publicly available LIDC/IDRI Database used in this study.

### 1.2 10-fold data splitting
We follow the official 10-fold data splitting from LUNA16 challenge and generate data split json files using the script from [nnDetection](https://github.com/MIC-DKFZ/nnDetection/blob/main/projects/Task016_Luna/scripts/prepare.py).

Please download the resulted json files from https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/LUNA16_datasplit-20220615T233840Z-001.zip.

In these files, the values of "box" are the ground truth boxes in world coordinate.

### 1.3 Data resampling
The raw CT images in LUNA16 have various of voxel sizes. The first step is to resample them to the same voxel size.
In this model, we resampled them into 0.703125 x 0.703125 x 1.25 mm.

Please following the instruction in Section 3.1 of https://github.com/Project-MONAI/tutorials/tree/main/detection to do the resampling.

Alternatively, we provide [resampled nifti images](https://drive.google.com/drive/folders/1JozrufA1VIZWJIc5A1EMV3J4CNCYovKK?usp=share_link) and a copy of [original mhd/raw images](https://drive.google.com/drive/folders/1-enN4eNEnKmjltevKg3W2V-Aj0nriQWE?usp=share_link) from [LUNA16](https://luna16.grand-challenge.org/Home/) for users to download.

## 2. Training configuration
The training was the following:

GPU: at least 16GB GPU memory

Actual Model Input: 192 x 192 x 80

AMP: True

Optimizer: Adam

Learning Rate: 1e-2

Loss: BCE loss and L1 loss

### Input
list of 1 channel 3D CT patches

### Output
In training mode: dictionary of classification and box regression loss in training mode;

In evaluation mode: list of dictionary of predicted box, classification label, and classification score in evaluation mode.

## 3. Performance
<p align="center">
  <img src="https://developer.download.nvidia.com/assets/Clara/Images/monai_retinanet_detection_train_and_val_metrics.png" alt="detection scheme")
</p>

With a single DGX1V 16G GPU, it took around 55 hours to train 300 epochs for each data fold. The pre-trained model was trained on fold 0.

The output of inference for each data fold is a result json file. The script to combine 10 result json files to one csv file can be found in https://github.com/Project-MONAI/tutorials/blob/main/detection/luna16_post_combine_cross_fold_results.py.
The script to compute FROC sensitivity value on 10-fold inference results can be downloaded from [LUNA16](https://luna16.grand-challenge.org/Evaluation/) in https://www.dropbox.com/s/wue67fg9bk5xdxt/evaluationScript.zip?dl=0. An example useage is in https://github.com/Project-MONAI/tutorials/blob/main/detection/run_luna16_offical_eval.sh.

This model achieves the following FROC sensitivity value on the 10-fold validation data:

| Methods             | 1/8   | 1/4   | 1/2   | 1     | 2     | 4     | 8     |
| :---:               | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [Liu et al. (2019)](https://arxiv.org/pdf/1906.03467.pdf)   | **0.848** | 0.876 | 0.905 | 0.933 | 0.943 | 0.957 | 0.970 |
| [nnDetection (2021)](https://arxiv.org/pdf/2106.00817.pdf)  | 0.812 | **0.885** | 0.927 | 0.950 | 0.969 | 0.979 | 0.985 |
| MONAI detection     | 0.835 | **0.885** | **0.931** | **0.957** | **0.974** | **0.983** | **0.988** |

**Table 1**. The FROC sensitivity values at the predefined false positive per scan thresholds of the LUNA16 challenge.


## 4. Commands example
Execute training:
```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf
```

Override the `train` config to execute evaluation with the trained model:
```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.json','configs/evaluate.json']" --logging_file configs/logging.conf
```

Execute inference on resampled LUNA16 images (resampled following Section 3.1 of https://github.com/Project-MONAI/tutorials/tree/main/detection) by setting `"whether_raw_luna16": false` in `inference.json`:
```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf
```
With the same command, we can execute inference on raw LUNA16 images by setting `"whether_raw_luna16": true` in `inference.json`. Remember to also set `"data_list_file_path": "$@bundle_root + '/LUNA16_datasplit/original/dataset_fold0.json'"` and change `"data_file_base_dir"`.

Note that in inference.json, the transform "LoadImaged" in "preprocessing" and "AffineBoxToWorldCoordinated" in "postprocessing" has `"affine_lps_to_ras": true`.
This depends on the input images. LUNA16 needs `"affine_lps_to_ras": true`.
It is possible that your inference dataset should set `"affine_lps_to_ras": false`.


# Disclaimer
This is an example, not to be used for diagnostic purposes.

# References
[1] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." ICCV 2017. https://arxiv.org/abs/1708.02002)

[2] Baumgartner and Jaeger et al. "nnDetection: A self-configuring method for medical object detection." MICCAI 2021. https://arxiv.org/pdf/2106.00817.pdf

[3] Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., Zhao, B., Aberle, D. R., Henschke, C. I., Hoffman, E. A., Kazerooni, E. A., MacMahon, H., Van Beek, E. J. R., Yankelevitz, D., Biancardi, A. M., Bland, P. H., Brown, M. S., Engelmann, R. M., Laderach, G. E., Max, D., Pais, R. C. , Qing, D. P. Y. , Roberts, R. Y., Smith, A. R., Starkey, A., Batra, P., Caligiuri, P., Farooqi, A., Gladish, G. W., Jude, C. M., Munden, R. F., Petkovska, I., Quint, L. E., Schwartz, L. H., Sundaram, B., Dodd, L. E., Fenimore, C., Gur, D., Petrick, N., Freymann, J., Kirby, J., Hughes, B., Casteele, A. V., Gupte, S., Sallam, M., Heath, M. D., Kuhn, M. H., Dharaiya, E., Burns, R., Fryd, D. S., Salganicoff, M., Anand, V., Shreter, U., Vastagh, S., Croft, B. Y., Clarke, L. P. (2015). Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX

[4] Armato SG 3rd, McLennan G, Bidaut L, McNitt-Gray MF, Meyer CR, Reeves AP, Zhao B, Aberle DR, Henschke CI, Hoffman EA, Kazerooni EA, MacMahon H, Van Beeke EJ, Yankelevitz D, Biancardi AM, Bland PH, Brown MS, Engelmann RM, Laderach GE, Max D, Pais RC, Qing DP, Roberts RY, Smith AR, Starkey A, Batrah P, Caligiuri P, Farooqi A, Gladish GW, Jude CM, Munden RF, Petkovska I, Quint LE, Schwartz LH, Sundaram B, Dodd LE, Fenimore C, Gur D, Petrick N, Freymann J, Kirby J, Hughes B, Casteele AV, Gupte S, Sallamm M, Heath MD, Kuhn MH, Dharaiya E, Burns R, Fryd DS, Salganicoff M, Anand V, Shreter U, Vastagh S, Croft BY.  The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. Medical Physics, 38: 915--931, 2011. DOI: https://doi.org/10.1118/1.3528204

[5] Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging, 26(6), 1045–1057. https://doi.org/10.1007/s10278-013-9622-7

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
