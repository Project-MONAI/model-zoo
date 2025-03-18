---
license: other
license_name: nvidia-oneway-noncommercial-license
license_link: LICENSE
tags:
- medical-imaging
- visual-language-model
- multimodal
- vila
- llama3
---

# Llama3-VILA-M3-3B

> Built with Meta Llama 3

## Model Overview

## Description:
M3 is a medical visual language model that empowers medical imaging professionals, researchers, and healthcare enterprises by enhancing medical imaging workflows across various modalities. 

Key features include:
- Integration with expert models from the MONAI Model Zoo
- Support for multiple imaging modalities
- Lightweight 3B parameter model for faster inference and reduced hardware requirements

For more details, see our [repo](https://github.com/Project-MONAI/VLM)

### Core Capabilities
M3 NIM provides a comprehensive suite of 2D medical image analysis tools, including:
1. Segmentation
2. Classification
3. Visual Question Answering (VQA)
4. Report/Findings Generation

These capabilities are applicable across various medical imaging modalities, leveraging expert models from the MONAI Model Zoo to ensure high-quality results.

## Model Architecture: 
**Architecture Type:** Auto-Regressive Vision Language Model
**Network Architecture:** [VILA](https://github.com/NVlabs/VILA) with Llama
**Parameters:** 3 billion parameters

## Input: 
**Input Type(s):** Text and Image
**Input Format(s):** Text: String, Image
**Input Parameters:** Text: 1D, Image: 2D 

## Output: 
**Output Type(s):** Text and Image 
**Output Format:** Text: String and Image
**Output Parameters:** Text: 1D, Image: 2D/3D 

## Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.