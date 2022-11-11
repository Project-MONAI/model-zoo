# MedNIST Hand Image Registration

Based on [the tutorial of 2D registration](https://github.com/Project-MONAI/tutorials/tree/main/2d_registration)

## Downloading the Dataset
Download the dataset [from here](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz) and extract the contents to a convenient location.

## Training

Training with same-subject image inputs
```bash
python -m monai.bundle run training --config_file configs/train.yaml
```

Training with cross-subject image inputs
```bash
python -m monai.bundle run training --config_file configs/train.yaml --cross_subjects true
```


## Inference

```bash
python -m monai.bundle run eval \
  --config_file configs/inference.yaml \
  --ckpt "models/model_key_metric=-0.0890.pt" \
  --logging_file configs/logging.conf \
  --device "cuda:1"
```
