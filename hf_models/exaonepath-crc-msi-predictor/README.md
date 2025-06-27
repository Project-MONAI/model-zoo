---
license: other
license_name: exaonepath
license_link: LICENSE
tags:
- lg-ai
- EXAONEPath-1.0
- pathology
- lg-ai
---

# EXAONEPath-CRC-MSI-Predictor

## MSI classification of CRC tumors
MSI classification of CRC tumors using EXAONEPath 1.0.0 Patch-level Foundation Model for Pathology.

[[`Paper`](https://arxiv.org/abs/2408.00380)] [[`Model`](https://huggingface.co/LGAI-EXAONE/EXAONEPath-CRC-MSI-Predictor/tree/main)] [[`BibTeX`](#citation)]

## Introduction
This model serves as a reference for predicting MSI status using CRC (colorectal cancer) tumor images as input. When the model receives an H&E-stained whole slide image as input, it removes artifacts observed in the image and extracts only tissue-related objects. These objects are then reconstructed into a set of tiles with a size of 256 by 256 pixels at an mpp (micron per pixel) of 0.5.

The tiles pass through the EXAONEPath v1.0 patch-level foundation model (https://huggingface.co/LGAI-EXAONE/EXAONEPath), which converts them into a set of features. These features are then integrated into a slide-level feature representation through an aggregator(see the figure below). Finally, a linear classifier predicts the MSI status (MSS or MSI-H/L).

The model achieves an average performance of AUROC 0.93 on TCGA-COAD + TCGA-READ data and 0.84 on in-house data.


## Quickstart

### Summary

1. Copy your WSI files in '''.svs''' format into the '''samples''' directory
2. Run inference

### 1. Hardware Requirements
- NVIDIA GPU is required
- Minimum 8GB GPU memory recommended
- NVIDIA driver version >= 450.80.02 required

### 2. Environment Setup
Create and activate a virutal environment.
```bash
python -m venv venv
source ./venv/bin/activate
```

Install huggingface_cli and download files
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download LGAI-EXAONE/EXAONEPath-CRC-MSI-Predictor --local-dir .
```

Install requirements
```bash
pip install -r requirements.txt
```

Verify pytorch with GPU support
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. Data Preparation
Copy your WSI files into the `samples` directory.

The program accepts ```.svs``` formatted files.   This format is used,
for example, for diagnostic slides as part of the TCGA-COAD project.  An
image from that project is available at the following link:

https://portal.gdc.cancer.gov/files/17cfcc8c-49a4-48ce-a5e1-4a3c582ce198

Download that data, extract the svs file from that compressed tar file, and
copy the svs file to the top level of the `samples` directory.

### 4. Inference
```bash
python -m monai.bundle run inference --meta_file configs/metadata.json --config_file configs/inference.yaml
```

### 5. Run-time errors

Particularly on Windows, if you receive the error
```
RuntimeError: Failed to evaluate ConfigExpression:
"$scripts.inference.infer(__local_refs['model'], __local_refs['input_files'])"
```
and references line 71 in the file "scripts/exaonepath.py"
```
    for count, patches in enumerate(patch_loader):
```
then you may have set the number of workers for the dataloader to 0.  This is
accomplished by changing line 65 of "scripts/exaonepath.py" to
```
            num_workers=0,
```
and removing lines 66 and 67, such that lines 62-67 become
```
        patch_loader = DataLoader(
            dataset=patch_dataset,
            batch_size=feature_extractor_batch_size,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
```

## License
The model is licensed under [EXAONEPath AI Model License Agreement 1.0 - NC](./LICENSE)

## Citation <a name="citation"></a>
If you find EXAONEPath useful, please cite it using this BibTeX:
```
@article{yun2024exaonepath,
  title={EXAONEPath 1.0 Patch-level Foundation Model for Pathology},
  author={Yun, Juseung and Hu, Yi and Kim, Jinhyung and Jang, Jongseong and Lee, Soonyoung},
  journal={arXiv preprint arXiv:2408.00380},
  year={2024}
}
```

## Contact
LG AI Research Technical Support: <a href="mailto:contact_us1@lgresearch.ai">contact_us1@lgresearch.ai</a>
