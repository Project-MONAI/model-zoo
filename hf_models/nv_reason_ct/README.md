# NV-Reason-CT

NV-Reason-CT is a generative vision-language model for native 3D chest and abdominal CT interpretation. It pairs a Primus 3D Vision Transformer (initialized from COLIPRI) with a Qwen3.5-4B language model and supports structured report generation, abnormality classification, visual question answering, and interactive multi-turn reasoning directly from NIfTI volumes.

| | |
|---|---|
| **Model weights** | [nvidia/NV-Reason-CT](https://huggingface.co/nvidia/NV-Reason-CT) |
| **Code (inference, SFT, GRPO)** | [NVIDIA-Medtech/NV-Reason-CT](https://github.com/NVIDIA-Medtech/NV-Reason-CT) |
| **Live demo** | [Hugging Face Space](https://huggingface.co/spaces/nvidia/nv-reason-ct) |
| **Paper** | [arXiv:2609.27511](https://arxiv.org/abs/2609.27511) |
| **Input** | One-channel chest or abdominal CT (`.nii` / `.nii.gz`, Hounsfield units) + text prompt |
| **Output** | Text: reports, answers, and reasoning traces |
| **License** | [OpenMDW-1.1](https://huggingface.co/nvidia/NV-Reason-CT/blob/main/LICENSE) |

## Highlights

- **Native 3D**: CT is resampled to 2 mm isotropic, anatomy-cropped to 192 x 192 x 192 voxels, and encoded as a 24 x 24 x 24 grid. All 13,824 visual tokens reach the language model without spatial merging, with 3D MRoPE position encoding.
- **Chest and abdomen**: select `anatomy_region="chest"` or `"abdomen"` and the processor applies the matching deterministic crop.
- **Reasoning-first training**: SFT on radiologist-authored and report-grounded reasoning, followed by GRPO with verifiable, region-aware abnormality rewards. Thinking can be disabled for concise answers.

On CT-RATE (18 labels), NV-Reason-CT reports a macro-F1 of 0.614 and macro-AUROC of 0.871 using a plain Yes/No prompt, with no classification head. See the model card for the full comparison.

## Quick start

The model uses custom 3D model and processor code hosted on Hugging Face, so `trust_remote_code=True` is required. A CUDA GPU is required.

```bash
pip install "torch>=2.9.0" "transformers==5.6.2" "monai>=1.5.1" "nibabel>=5.3.3" \
    "dynamic-network-architectures==0.4.3" scipy timm einops safetensors
```

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "nvidia/NV-Reason-CT"
model = AutoModelForImageTextToText.from_pretrained(
    model_id, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
).eval().to("cuda")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "write a structured chest CT report"},
]}]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
inputs = processor(
    text=prompt, images3d=["chest_ct.nii.gz"], anatomy_region="chest", return_tensors="pt"
).to(model.device)

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
print(processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0])
```

The [GitHub repository](https://github.com/NVIDIA-Medtech/NV-Reason-CT) includes a command-line `inference.py`, multi-turn follow-up examples, example CT volumes, and SFT and GRPO fine-tuning recipes.

## Intended use

NV-Reason-CT is for research and development only. It is not a medical device and must not be used as a substitute for professional clinical judgment, diagnosis, or treatment decisions. Model outputs, including generated reasoning, can be incorrect and should be reviewed by qualified medical professionals.

## License

The model weights and code are released under the [OpenMDW-1.1 license](https://huggingface.co/nvidia/NV-Reason-CT/blob/main/LICENSE).

## Citation

```bibtex
@article{nvreasonct2026,
  title={NV-Reason-CT: 3D Visual Language Model for CT Analysis},
  journal={arXiv preprint arXiv:2609.27511},
  year={2026}
}
```
