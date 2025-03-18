---
license: cc-by-nc-sa-4.0
tags:
- computed-tomography
- chest-ct
- medical-imaging
- dataset
- multimodal
---

# CT_RATE Dataset

## [Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography](https://arxiv.org/abs/2403.17834)

CT_RATE is a pioneering dataset in 3D medical imaging that uniquely pairs textual data with image data focused on chest CT volumes. The dataset comprises chest CT volumes paired with corresponding radiology text reports, multi-abnormality labels, and metadata, all freely accessible to researchers.

## Dataset Overview

CT_RATE consists of 25,692 non-contrast chest CT volumes, expanded to 50,188 through various reconstructions, from 21,304 unique patients, along with corresponding radiology text reports, multi-abnormality labels, and metadata.

The dataset is divided into:
- Training set: 20,000 patients
- Validation set: 1,304 patients

File naming convention: `split_patientID_scanID_reconstructionID`  
For example, "valid_53_a_1" indicates a CT volume from the validation set, scan "a" from patient 53, and reconstruction 1 of scan "a".

## Applications

This dataset has been used to develop several groundbreaking models:

### CT-CLIP
A CT-focused contrastive language-image pre-training framework. As a versatile, self-supervised model, CT-CLIP is designed for broad application and does not require task-specific training. CT-CLIP outperforms state-of-the-art, fully supervised methods in multi-abnormality detection across all key metrics.

### CT-CHAT
A multimodal AI assistant designed to enhance the interpretation and diagnostic capabilities of 3D chest CT imaging. Building on CT-CLIP, it integrates both visual and language processing to handle diverse tasks like visual question answering, report generation, and multiple-choice questions.

## Dataset Configurations

1. **Labels**: Multi-abnormality labels for the CT volumes
2. **Reports**: Corresponding radiology text reports
3. **Metadata**: Additional metadata for each CT volume

## Terms and Conditions

Users of the CT_RATE dataset must agree to the [Terms and Conditions](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) which specify:

- The dataset is intended solely for academic, research, and educational purposes
- Any commercial exploitation is forbidden without permission
- Users must maintain data confidentiality and comply with data protection laws
- Proper attribution is required in any publications resulting from dataset use
- Redistribution of the dataset is not allowed

## Ethical Approval

Ethical approval documentation is available for researchers who require it for grant applications.

## Citation

When using this dataset, please consider citing the following related papers:

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