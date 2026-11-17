# NV-Reason-CXR-3B

NV-Reason-CXR-3B is a vision-language model for chest X-ray interpretation that explains its conclusions. It responds in the style of a senior radiologist teaching a case: a systematic, step-by-step review of the image, then findings, differential diagnoses, and recommendations. It also supports follow-up questions such as "provide differentials" or "write a structured report".

| | |
|---|---|
| **Model weights** | [nvidia/NV-Reason-CXR-3B](https://huggingface.co/nvidia/NV-Reason-CXR-3B) |
| **Code (SFT, GRPO)** | [NVIDIA-Medtech/NV-Reason-CXR](https://github.com/NVIDIA-Medtech/NV-Reason-CXR) |
| **Live demo** | [Hugging Face Space](https://huggingface.co/spaces/nvidia/nv-reason-cxr) |
| **Paper** | [arXiv:2510.23968](https://arxiv.org/abs/2510.23968) |
| **Base model** | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| **Input** | Frontal chest X-ray (JPEG / PNG) + text prompt |
| **Output** | Text: reasoning trace, findings, differentials, structured report |
| **License** | [NVIDIA OneWay Noncommercial License](https://huggingface.co/nvidia/NV-Reason-CXR-3B/blob/main/LICENSE) |

## Highlights

- **Chain-of-thought for CXR**: the model learns how radiologists reason, not only what they conclude. Training data includes radiologist think-aloud annotations captured by voice and transcribed.
- **SFT then GRPO**: a radiologist-style supervised warm start, followed by reinforcement learning with verifiable rewards over a list of chest X-ray abnormalities.
- **Reader study**: in a preliminary study with US board-certified radiologists, full AI reasoning increased reader confidence and reduced time to finalize reports, particularly for abnormal cases.

## Quick start

```bash
pip install torch==2.7.1 torchvision==0.22.1 transformers==4.56.1
```

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

model_name = "nvidia/NV-Reason-CXR-3B"
model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch.float16).eval().to("cuda")
processor = AutoProcessor.from_pretrained(model_name)

image = Image.open("chest_xray.png")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Find abnormalities and support devices."},
]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=2048)
trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
print(processor.batch_decode(trimmed, skip_special_tokens=True)[0])
```

The [GitHub repository](https://github.com/NVIDIA-Medtech/NV-Reason-CXR) provides SFT and GRPO training examples (8 x A100 80GB) using MIMIC-CXR and CheXpert subsets.

## Intended use

NV-Reason-CXR-3B is for research and educational purposes only and should not be used for clinical diagnosis or treatment decisions. All outputs should be reviewed by qualified medical professionals.

## License

The model weights are released under the [NVIDIA OneWay Noncommercial License](https://huggingface.co/nvidia/NV-Reason-CXR-3B/blob/main/LICENSE). The training code in the GitHub repository is released under Apache 2.0.

## Citation

```bibtex
@misc{myronenko2025reasoning,
  title={Reasoning Visual Language Model for Chest X-Ray Analysis},
  author={Andriy Myronenko and Dong Yang and Baris Turkbey and Mariam Aboian and Sena Azamat and Esra Akcicek and Hongxu Yin and Pavlo Molchanov and Marc Edgar and Yufan He and Pengfei Guo and Yucheng Tang and Daguang Xu},
  year={2025},
  eprint={2510.23968},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2510.23968}
}
```
