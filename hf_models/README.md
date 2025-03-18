# Hugging Face Models

This directory contains models that are hosted on Hugging Face. **Important: These models do not follow the traditional MONAI Bundle format and cannot be run using the standard MONAI Bundle APIs.**

Each model directory contains:

1. `metadata.json` - Model metadata following a similar schema to MONAI Bundles
2. `README.md` - Detailed documentation about the model
3. `LICENSE` - Model license

## Using HF Models

These models must be accessed directly from Hugging Face using the `huggingface_hub` and `transformers` libraries. For complete usage instructions and examples, please visit the corresponding Hugging Face model repository linked below.

### Available Models

| Model | Description | HF Repository |
|-------|-------------|--------------|
| exaonepath | EXAONEPath is a patch-level pathology pretrained model with 86 million parameters | [LGAI-EXAONE/EXAONEPath](https://huggingface.co/LGAI-EXAONE/EXAONEPath) |
| llama3_vila_m3_3b | Lightweight medical vision language model that enhances VLMs with medical expert knowledge (3B parameters) | [MONAI/Llama3-VILA-M3-3B](https://huggingface.co/MONAI/Llama3-VILA-M3-3B) |
| llama3_vila_m3_8b | Medical vision language model that utilizes domain-expert models to improve precision in medical imaging tasks (8B parameters) | [MONAI/Llama3-VILA-M3-8B](https://huggingface.co/MONAI/Llama3-VILA-M3-8B) |
| llama3_vila_m3_13b | Enhanced medical vision language model with improved capabilities for various medical imaging tasks (13B parameters) | [MONAI/Llama3-VILA-M3-13B](https://huggingface.co/MONAI/Llama3-VILA-M3-13B) |
| ct_chat | Vision-language foundational chat model for 3D chest CT volumes | [ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) |
