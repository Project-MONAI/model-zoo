---
license: cc-by-nc-sa-4.0
tags:
- computed-tomography
- chest-ct
- medical-imaging
- vision-language-model
- multimodal
- medical-assistant
---

# CT-CHAT Model

## [Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography](https://arxiv.org/abs/2403.17834)

## Model Overview

CT-CHAT is a vision-language foundational chat model for 3D chest CT volumes. Leveraging the VQA dataset derived from CT-RATE and pretrained 3D vision encoder from CT-CLIP, we developed this multimodal AI assistant specifically designed to enhance the interpretation and diagnostic capabilities of 3D chest CT imaging.

Building on the strong foundation of CT-CLIP, CT-CHAT integrates both visual and language processing to handle diverse tasks including:
- Visual question answering
- Radiology report generation
- Multiple-choice diagnostic questions

Trained on over 2.7 million question-answer pairs from the CT-RATE dataset, CT-CHAT leverages 3D spatial information, making it superior to 2D-based models. The model not only improves radiologist workflows by reducing interpretation time but also delivers highly accurate and clinically relevant responses, pushing the boundaries of 3D medical imaging analysis.

## Technical Foundation

CT-CHAT builds upon two key technological innovations:

### CT-CLIP
A CT-focused contrastive language-image pre-training framework that serves as the visual encoder for CT-CHAT. As a versatile, self-supervised model, CT-CLIP is designed for broad application and outperforms state-of-the-art, fully supervised methods in multi-abnormality detection.

### CT-RATE Dataset
A pioneering dataset of 25,692 non-contrast chest CT volumes (expanded to 50,188 through various reconstructions) paired with corresponding radiology text reports, multi-abnormality labels, and metadata from 21,304 unique patients.

## Model Capabilities

1. **Visual Question Answering**: Answer free-form questions about 3D CT volumes
2. **Report Generation**: Create comprehensive radiology reports from CT scans
3. **Diagnostic Support**: Assist with differential diagnoses and abnormality detection
4. **Educational Use**: Train medical students and residents on CT interpretation

## Terms and Conditions

Users of the CT-CHAT model must agree to the [Terms and Conditions](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) which specify:

- The model is intended solely for academic, research, and educational purposes
- Any commercial exploitation is forbidden without permission
- Users must maintain data confidentiality and comply with data protection laws
- Proper attribution is required in any publications resulting from model use
- Redistribution of the model is not allowed

## Citation

When using this model, please consider citing the following related papers:

```bibtex
@misc{hamamci2024foundation,
      title={Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography},
      author={Ibrahim Ethem Hamamci and Sezgin Er and Furkan Almas and Ayse Gulnihan Simsek and Sevval Nil Esirgun and Irem Dogan and Muhammed Furkan Dasdelen and Omer Faruk Durugol and Bastian Wittmann and Tamaz Amiranashvili and Enis Simsar and Mehmet Simsar and Emine Bensu Erdemir and Abdullah Alanbay and Anjany Sekuboyina and Berkan Lafci and Christian Bluethgen and Mehmet Kemal Ozdemir and Bjoern Menze},
      year={2024},
      eprint={2403.17834},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.17834},
}

@misc{hamamci2024generatect,
      title={GenerateCT: Text-Conditional Generation of 3D Chest CT Volumes},
      author={Ibrahim Ethem Hamamci and Sezgin Er and Anjany Sekuboyina and Enis Simsar and Alperen Tezcan and Ayse Gulnihan Simsek and Sevval Nil Esirgun and Furkan Almas and Irem Dogan and Muhammed Furkan Dasdelen and Chinmay Prabhakar and Hadrien Reynaud and Sarthak Pati and Christian Bluethgen and Mehmet Kemal Ozdemir and Bjoern Menze},
      year={2024},
      eprint={2305.16037},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2305.16037},
}

@misc{hamamci2024ct2rep,
      title={CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging},
      author={Ibrahim Ethem Hamamci and Sezgin Er and Bjoern Menze},
      year={2024},
      eprint={2403.06801},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2403.06801},
}
```
