# Coronsegmentator

### **Authors**

Y. Ke, MC. Chen, TY. Lin, YC. Chan Foxconn Digital Health AI Team

### **Tags**

Segmentation, CT, Heart, Coronary Artery, USD, MONAI, nnU-Net, 3D Reconstruction

## **Model Description**

Coronsegmentator is an automated pipeline that performs coronary artery segmentation on cardiac CT images using a pretrained nnU-Net model. The pipeline converts segmentation results (in NIfTI format) into STL files for downstream 3D visualization and digital twin simulation using NVIDIA Omniverse.

The workflow is designed to enable precise, scalable, and automated generation of personalized coronary artery digital twins, supporting clinical planning, AI-based diagnosis, and medical visualization in research or pre-operative planning workflows.

## **Data**

The Coronsegmentator model was trained on ImageCAS, a dataset consisting of annotated images for training/validation respectively.

The input data consists of anonymized 3D CT scans in .nii.gz format. This pipeline was designed to be compatible with preoperative cardiac CT data, including varying scanner vendors (e.g., Siemens, Philips).

Each image is processed by:

1. Segmenting coronary arteries using a pretrained nnU-Net model

2. Saving segmentation results as STL files

#### **Preprocessing**

Input: .nii.gz NIfTI format (CT scan)

Resolution normalization handled internally

No manual annotation required for inference

#### **Inference**

```bash
python -m monai.bundle run \
  --config_file configs/inference.json \
  --meta_file configs/metadata.json \
  --logging_file configs/logging.conf \
  --bundle_root <path/to/CoronSegmentator> \
  --dataset_dir <path/to/nifti_inputs>
```

The ImageCAS dataset is publicly available at:
https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT.git

## **Limitations**

This model is intended for research use only. It has not been validated for clinical deployment and should not be used for patient treatment decisions.

## **References**

[1] Zeng, An, et al. "ImageCAS: A large-scale dataset and benchmark for coronary artery segmentation based on computed tomography angiography images." Computerized Medical Imaging and Graphics 109 (2023): 102287.

[2] Isensee, Fabian, et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature methods 18.2 (2021): 203-211.

## **License**

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
