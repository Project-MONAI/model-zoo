# Hugging Face Models

This directory contains models that are hosted on Hugging Face. **Important: These models do not follow the traditional MONAI Bundle format and cannot be run using the standard MONAI Bundle APIs.**

Each model directory contains:

1. `configs/metadata.json` - Model metadata following a similar schema to MONAI Bundles
2. `configs/inference.json` - Configuration that references the HF model but may not be directly executable
3. `configs/logging.conf` - Logging configuration
4. `docs/README.md` - Detailed documentation about the model
5. `large_files.yml` - References the Hugging Face model repository
6. `LICENSE` - Model license

## Using HF Models

These models must be accessed directly from Hugging Face using the `huggingface_hub` and `transformers` libraries. For complete usage instructions and examples, please visit the corresponding Hugging Face model repository linked below.

### Authentication

Some models may require authentication with a Hugging Face token. You can set your token as an environment variable:

```bash
export HF_TOKEN=your_huggingface_token
```

### Available Models

| Model | Description | HF Repository |
|-------|-------------|--------------|
| exaonepath | EXAONEPath is a patch-level pathology pretrained model with 86 million parameters | [LGAI-EXAONE/EXAONEPath](https://huggingface.co/LGAI-EXAONE/EXAONEPath) |
| llama3-vila-m3-3b | Lightweight medical visual language model based on VILA and Llama 3 (3B parameters) | [MONAI/Llama3-VILA-M3-3B](https://huggingface.co/MONAI/Llama3-VILA-M3-3B) |
| llama3-vila-m3-8b | Medical visual language model based on VILA and Llama 3 that supports medical image analysis | [MONAI/Llama3-VILA-M3-8B](https://huggingface.co/MONAI/Llama3-VILA-M3-8B) |
| llama3-vila-m3-13b | Enhanced medical visual language model based on VILA and Llama 3 with improved reasoning capabilities (13B parameters) | [MONAI/Llama3-VILA-M3-13B](https://huggingface.co/MONAI/Llama3-VILA-M3-13B) |
| ct-rate | Pioneering dataset of chest CT volumes paired with radiology reports, multi-abnormality labels, and metadata | [ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) |