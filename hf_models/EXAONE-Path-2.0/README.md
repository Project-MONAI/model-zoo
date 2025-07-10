---
license: other
license_name: exaonepath
license_link: LICENSE
tags:
- lg-ai
- EXAONE-Path-2.0
- pathology
---

# EXAONE Path 2.0

## Introduction
In digital pathology, whole-slide images (WSIs) are often difficult to handle due to their gigapixel scale, so most approaches train patch encoders via self-supervised learning (SSL) and then aggregate the patch-level embeddings via multiple instance learning (MIL) or slide encoders for downstream tasks.
However, patch-level SSL may overlook complex domain-specific features that are essential for biomarker prediction, such as mutation status and molecular characteristics, as SSL methods rely only on basic augmentations selected for natural image domains on small patch-level area.
Moreover, SSL methods remain less data efficient than fully supervised approaches, requiring extensive computational resources and datasets to achieve competitive performance.
To address these limitations, we present EXAONE Path 2.0, a pathology foundation model that learns patch-level representations under direct slide-level supervision.
Using only 35k WSIs for training, EXAONE Path 2.0 achieves state-of-the-art average performance across 10 biomarker prediction tasks, demonstrating remarkable data efficiency.
For further details, please refer to EXAONE_Path_2_0_technical_report.pdf.

## Quickstart
Load EXAONE Path 2.0 and extract features.

### 1. Prerequisites ###
- NVIDIA GPU with 12GB+ VRAM
- Python 3.12+

Note: This implementation requires NVIDIA GPU and drivers. The provided environment setup specifically uses CUDA-enabled PyTorch, making NVIDIA GPU mandatory for running the model.

### 2. Setup Python environment ###
```bash
git clone https://github.com/LG-AI-EXAONE/EXAONE-Path-2.0.git
cd EXAONE-Path-2.0
pip install -r requirements.txt
```

### 3. Load the model & Inference
```python
from exaonepath import EXAONEPathV20

hf_token = "YOUR_HUGGING_FACE_ACCESS_TOKEN"
model = EXAONEPathV20.from_pretrained("LGAI-EXAONE/EXAONE-Path-2.0", use_auth_token=hf_token)

svs_path = "samples/sample.svs"
patch_features = model(svs_path)[0]
```

## Model Performance Comparison

Performance of EXAONE Path 2.0 on 10 slide-level benchmarks (AUROC scores):

| **Benchmarks** | **TITAN** | **PRISM** | **CHIEF** | **Prov-GigaPath** | **UNI2-h** | **EXAONE Path 1.0** | **EXAONE Path 2.0** |
|---|---|---|---|---|---|---|---|
| LUAD-TMB-USA1 | 0.690 | 0.645 | 0.650 | 0.674 | 0.669 | 0.692 | 0.664 |
| LUAD-EGFR-USA1 | 0.754 | 0.815 | 0.784 | 0.709 | 0.827 | 0.784 | 0.853 |
| LUAD-KRAS-USA2 | 0.541 | 0.623 | 0.468 | 0.511 | 0.469 | 0.527 | 0.645 |
| CRC-MSI-KOR | 0.937 | 0.943 | 0.927 | 0.954 | 0.981 | 0.972 | 0.938 |
| BRCA-TP53-CPTAC | 0.788 | 0.842 | 0.788 | 0.739 | 0.808 | 0.766 | 0.757 |
| BRCA-PIK3CA-CPTAC | 0.758 | 0.893 | 0.702 | 0.735 | 0.857 | 0.735 | 0.804 |
| RCC-PBRM1-CPTAC | 0.638 | 0.557 | 0.513 | 0.527 | 0.501 | 0.526 | 0.583 |
| RCC-BAP1-CPTAC | 0.719 | 0.769 | 0.731 | 0.697 | 0.716 | 0.719 | 0.807 |
| COAD-KRAS-CPTAC | 0.764 | 0.744 | 0.699 | 0.815 | 0.943 | 0.767 | 0.912 |
| COAD-TP53-CPTAC | 0.889 | 0.816 | 0.701 | 0.712 | 0.783 | 0.819 | 0.875 |
| **Average** | 0.748 | 0.765 | 0.696 | 0.707 | 0.755 | 0.731 | **0.784** |

<br>


## License
The model is licensed under [EXAONEPath AI Model License Agreement 1.0 - NC](./LICENSE)

<!-- ## Citation
If you find EXAONE Path 2.0 useful, please cite it using this BibTeX:
```
@article{yun2024exaonepath,
  title={EXAONE Path 2.0 Techincal Report},
  author={Yun, Juseung and Hu, Yi and Kim, Jinhyung and Jang, Jongseong and Lee, Soonyoung},
  journal={arXiv preprint arXiv:2408.00380},
  year={2024}
}
``` -->

## Contact
LG AI Research Technical Support: <a href="mailto:contact_us1@lgresearch.ai">contact_us1@lgresearch.ai</a>
