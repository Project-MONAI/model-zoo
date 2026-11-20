# Breast lesion MoE classification

### **Authors**
Ilyass Emssaad

### **Tags**
Classification, Pathology, Breast, Histopathology, Mixture-of-Experts, BRACS

## **Model Description**
This bundle ships a trained mixture-of-experts (MoE) head that classifies breast
lesions in H&E histopathology as **benign / atypical / malignant**.

Unlike image-input bundles, this bundle takes **pre-computed pathology
foundation-model embeddings** as input -- one float32 vector per expert for each
H&E tile / ROI:

- **UNI** -- 1024-d (CLS token)
- **Virchow2** -- 2560-d (CLS token + mean patch token)
- **Phikon-v2, domain-adapted (fine-tuned)** -- 1024-d (CLS token)

Each expert embedding is linearly projected to a shared 512-d space (with
LayerNorm); a learned **soft top-2 router** (Switch-style, dropless, with a
load-balance auxiliary loss during training) mixes the experts, and a linear
classifier maps the routed embedding to 3-class logits. The router's gate
weights are exposed by the underlying head for auditability (which expert the
model listened to, per case), which the accompanying agentic system uses as
expert-attribution evidence.

The shipped weights (`models/model.pt`, 2,757,638 parameters, ~11 MB) are the
**single-seed (seed 0)** MoE head trained on the BRACS official train split.
The network code in `scripts/moe_network.py` is vendored from
https://github.com/eliem08/paper_code (`src/breastpath_copilot/models/moe.py`).

The bundle does not support TorchScript export: the network forward takes a
mapping of embedding tensors (and an optional magnification list), which
TorchScript cannot trace. It also does not support TensorRT export.

## **Data**
The model was trained and evaluated on **BRACS** (BReAst Carcinoma Subtyping,
https://www.bracs.icar.cnr.it), using the official patient-disjoint
train/val/test split. The BRACS test split used for the headline results has
**570 ROIs** (242 benign, 162 atypical, 166 malignant); the retrieval index
shipped with the paper code covers 3,657 BRACS train cases.

No images and no datasets are redistributed in this bundle. To embed new tiles
you need the datasets from their maintainers (BRACS, plus BACH and BreaKHis,
which the paper uses as out-of-distribution / cross-cohort evaluation) and the
foundation models from Hugging Face -- UNI and Virchow2 are gated (accept their
terms, set `HF_TOKEN`), and the domain-adapted Phikon-v2 expert weights are
**not currently public** (see Limitations). Embedding extraction is
`scripts/extract_fm_embeddings.py` in the paper-code repository. See
`docs/data_license.txt` for the full terms.

#### **Preprocessing**
There is no image preprocessing in this bundle: inputs are already embedding
vectors. The dataset class (`scripts/embedding_dataset.py`) loads `.npy`,
`.pt`, `.json` or `.csv` float32 vectors from a datalist JSON (see Commands
below) and checks each vector against its expected dim (1024 / 2560 / 1024).

#### **Sample data**
`sample_data/` holds **synthetic** embeddings (seeded normal vectors with the
observed embedding scale, stored as `.json`, 4 cases) and
`configs/sample_datalist.json` points at them. They exist only so the
inference pipeline can be smoke-tested end to end; predictions on them are
meaningless. Do not mistake them for real BRACS embeddings.

## **Performance**
Headline results are the paper's **5-seed deep-ensemble** results (soft top-2
routing, shared dim 512) on the BRACS official test split (570 ROIs), from
`artifacts/results/moe_perclass_3class.json` in the paper-code repository:

| Metric | 3-class ensemble |
|---|---|
| Accuracy | 0.8596 |
| Weighted F1 | 0.859 |
| Macro-AUROC | 0.9578 |
| ECE (temperature 2.631) | 0.104 calibrated (0.122 uncalibrated) |

Per-class F1 (ensemble): benign 0.876, atypical 0.772, malignant 0.918.
Per-seed weighted F1 across the 5 seeds: 0.8819, 0.8536, 0.8519, 0.8489, 0.8533.

> **Honest status of the shipped checkpoint.** The numbers above describe the
> 5-seed ensemble, not the single-seed checkpoint in this bundle. The shipped
> checkpoint is seed 0 of that ensemble; its own test metrics have not been
> re-measured in isolation yet and are **to be confirmed** -- the ensemble
> numbers are the correct headline for the method, not a claim about this one
> file. A 7-class head (N, PB, UDH, FEA, ADH, DCIS, IC) was also trained in the
> paper (ensemble weighted F1 0.726, macro-AUROC 0.9406) but is **not** shipped
> here; this bundle is 3-class only.

Cross-cohort evidence from the paper (same MoE, frozen scheme): BACH 4-class
5-fold cross-validation accuracy 0.9625, weighted F1 0.9623, macro-AUROC
0.9977. Conformal prediction (3-class, target coverage 90%): empirical
coverage 0.9596, mean set size 1.377.

## **Commands Example**
Extract embeddings for your own tiles first (outside this bundle):

```
# in https://github.com/eliem08/paper_code
python scripts/extract_fm_embeddings.py --help
```

Build a datalist from a directory of `<image_id>__<expert>.npy` files:

```
python scripts/create_datalist.py --embeddings_dir <dir with .npy files> \
    --output_file configs/my_datalist.json --labels_csv <optional image_id,label CSV>
```

Add the scripts folder to your python path, as in other bundles:

```
export PYTHONPATH=$PYTHONPATH:<path to the bundle root dir>/scripts
```

Download the weights (listed in `large_files.yml`; MONAI downloads them
automatically on `monai.bundle download`, or fetch the file directly):

```
python -m monai.bundle download --name breast_lesion_moe_classification --bundle_dir <dir>
```

Execute inference (point `datalist_file` at your own datalist with the
`--datalist_file <path>` override, or edit `configs/inference.json`):

```
python -m monai.bundle run run --meta_file configs/metadata.json \
    --config_file configs/inference.json configs/logging.conf
```

Predictions are written to `output/predictions.csv` with one row per case
(`image_id`, `benign`, `atypical`, `malignant` probabilities).

## **Execute training**
Retrain the MoE head from cached embeddings with `configs/train.json`
(120 epochs, AdamW lr 1e-3, cross-entropy, seed 0 -- the paper's recipe in
`scripts/train_agent_moe_ckpt.py` of the paper-code repository):

```
python -m monai.bundle run run --meta_file configs/metadata.json \
    --config_file configs/train.json configs/logging.conf
```

Set `dataset_dir` in `train.json` to the directory holding
`train_datalist.json` / `val_datalist.json` (datalists of cached-embedding
paths, same format as above). Two documented simplifications vs the paper
recipe: this config uses plain cross-entropy (the paper adds a 0.01
load-balance auxiliary loss) and no class weights (the paper uses
train-split class weights). For the exact paper recipe, run
`scripts/train_agent_moe_ckpt.py` in the paper-code repository.

## **System Configuration**
MoE-head training on cached embeddings takes minutes on a single GPU (per the
paper-code README); embedding extraction needs a CUDA GPU and the gated
foundation-model weights. Inference on the 2.8M-parameter head runs on CPU
(the full inference pipeline was smoke-tested on CPU with MONAI 1.4.0 /
PyTorch 2.5.1).

## **Limitations**
- **Research use only.** This bundle is not cleared or approved by the FDA or
  any regulatory agency and cannot be used for clinical procedures or
  diagnostic decisions.
- **Embeddings in, not images.** The bundle cannot classify a raw H&E image;
  it needs UNI / Virchow2 / Phikon-v2 embeddings produced beforehand.
- **The domain-adapted Phikon-v2 expert weights are not currently public.**
  Until they are published, new tiles can only be embedded with the frozen
  Phikon-v2 model, which does not match the embeddings this head was trained
  on (the head expects `phikon_v2_ft` embeddings). Publishing those weights
  (or a frozen-Phikon-v2 head) is the remaining gap for fully reproducible
  end-to-end inference.
- **Single-checkpoint metrics to be confirmed** (see Performance).
- **Noncommercial license.** The bundle (code, weights, docs) is under the
  PolyForm Noncommercial License 1.0.0 -- noncommercial use only; commercial
  use needs a separate license from the authors. This is not the Apache-2.0
  license most zoo bundles use; see `LICENSE`.

## **Citation Info**
```
Toward reliable early breast lesion detection: a state-of-the-art,
uncertainty-aware diagnostic engine with agentic decision support in H&E
histopathology (under review).
Code, weights and results: https://github.com/eliem08/paper_code
BRACS dataset: https://www.bracs.icar.cnr.it
```

## **Contributors**
Ilyass Emssaad. Questions: emssaadelias@gmail.com. Code and results:
https://github.com/eliem08/paper_code

## **License**
Copyright (c) 2026 The Authors. Licensed under the PolyForm Noncommercial
License 1.0.0 -- see `LICENSE` in the bundle root.
