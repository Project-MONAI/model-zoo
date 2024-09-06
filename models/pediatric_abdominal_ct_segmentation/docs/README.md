# Model Overview
A Pediatric 3D Abdominal Organ Segmentation model, pretrained on adult and pediatric public datasets, and fine tuned for institutional pediatric data.

Please cite this manuscript:
Somasundaram E, Taylor Z, Alves VV, et al. Deep-Learning Models for Abdominal CT Organ Segmentation in Children: Development and Validation in Internal and Heterogeneous Public Datasets. AJR 2024 May 1 [published online]. Accepted manuscript. doi:10.2214/AJR.24.30931

## Data
Modality:
- CT

Organs Segmented:
- Liver
- Spleen
- Pancreas

Pre-training data:
- Total Segmentator (815)
- BTCV (30)
- TCIA Pediatric (282)

Fine-tuning data:
- Cincinnati Children's Liver Spleen CT dataset (275)
- Cincinnati Children's Pancreas CT dataset (146)

Testing data:
- Cincinnati Children's Liver-Spleen (57)
- Cincinnati Children's Pancreas (35)
- TCIA-Pediatric (74)
- Total Segmentator (50)

External dataset licenses can be found in accompanying text file. Internal datasets currently not publicly available.

To load data for training / inference / evaluate:

Ensure that the "image" and "label" parameters within the "training" and "validation" sections in configs/TS_test.json (or a new dataset json), as well as the "datalist" and "dataset_dir" in configs/train.yaml, configs/inference.yaml, and configs/evaluate-standalone.yaml files (or the according yaml if using multigpu / parallel or different model inferencing) are each changed to match the intended dataset's values.

One may make separate .json files detailing which exam images / masks are to be used in the same format as configs/TS_test.json with "training" and "validation" under root, as long as the "datalist_file_path" and "dataset_dir" values is changed accordingly in configs/train.yaml and configs/inference.yaml, and configs/evaluate-standalone.yaml (or the according yaml in different circumstances).

Ensure data folder structure is as follows, with scan files in the primary dataset folder, and mask files in the /labels/final subfolder:
    dataset/
    ├─ exam_001.nii.gz
    ├─ exam_002.nii.gz
    ├─ ...
    ├─ labels/
    │  ├─ final/
    │  │  ├─ exam_001.nii.gz
    │  │  ├─ exam_002.nii.gz
    │  │  ├─ ...

Configuration defaults are currently set to the external TotalSegmentator CT dataset.

### Model Architectures
- DynUNet
- SegResNet
- SwinUNETR

### Hyper-Parameter Tuning
Weights and Biases was used to extensively tune each model for learning rate, scheduler and optimizer. For fine-tuning the fraction of trainable layers was also optimized. DynUNet performed overall better on all test datasets. The Total Segmentator model was also compared and the DynUNet model significantly outperformed Total Segmentator on institutional test data while maintaining relatively stable performance on adult and TCIA datasets.

### Input
One channel CT image

### Output
Four channel CT label
- Label 3: pancreas
- Label 2: spleen
- Label 1: liver
- Label 0: background
- 96x96x96

## Performance
 - MedArxiv to be linked


## MONAI Bundle Commands
In addition to the Pythonic APIs, a few command line interfaces (CLI) are provided to interact with the bundle. The CLI supports flexible use cases, such as overriding configs at runtime and predefining arguments in a file.

For more details usage instructions, visit the [MONAI Bundle Configuration Page](https://docs.monai.io/en/latest/config_syntax.html).


#### Execute training:
Dataset used defaults to TotalSegmentator (https://zenodo.org/records/6802614#.ZFPll4TMKUk)
```
python -m monai.bundle run --config_file configs/train.yaml
```

Please note that if the default dataset path is not modified with the actual path in the bundle config files, you can also override it by using `--dataset_dir`:

```
python -m monai.bundle run --config_file configs/train.yaml --dataset_dir <actual dataset path>
```

#### `train` config to execute multi-GPU training:

```
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run --config_file configs/train-multigpu.yaml
```

#### Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run --config_file "['configs/train.yaml','configs/evaluate.yaml']"
```

#### Execute inference:

```
python -m monai.bundle run --config_file configs/inference.yaml
```

#### Execute standalone `evaluate`:
```
python -m monai.bundle run --config_file configs/evaluate.yaml
```


#### Execute standalone `evaluate` in parallel:
```
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run --config_file configs/evaluate-standalone.yaml
```


#### Export checkpoint for TorchScript:

```
python -m monai.bundle ckpt_export network_def --filepath models/dynunet_FT.ts --ckpt_file models/dynunet_FT.pt --meta_file configs/metadata.json --config_file configs/inference.yaml
```

#### Export checkpoint to TensorRT based models with fp32 or fp16 precision:

```
python -m monai.bundle trt_export --net_id network_def --filepath models/A100/dynunet_FT_trt_16.ts --ckpt_file models/dynunet_FT.pt --meta_file configs/metadata.json --config_file configs/inference.yaml  --precision <fp32/fp16> --use_trace "True" --dynamic_batchsize "[1, 4, 8]" --converter_kwargs "{'truncate_long_and_double':True, 'torch_executed_ops': ['aten::upsample_trilinear3d']}"
```

#### Execute inference with the TensorRT model:

```
python -m monai.bundle run --config_file "['configs/inference.yaml', 'configs/inference_trt.yaml']"
```

# References

[1] Somasundaram E, Taylor Z, Alves VV, et al. Deep-Learning Models for Abdominal CT Organ Segmentation in Children: Development and Validation in Internal and Heterogeneous Public Datasets. AJR 2024 May 1 [published online]. Accepted manuscript. doi:10.2214/AJR.24.30931

[2] Wasserthal, J., Breit, H.-C., Meyer, M. T., Pradella, M., Hinck, D., Sauter, A. W., Heye, T., Boll, D., Cyriac, J., Yang, S., Bach, M., & Segeroth, M. (2023, June 16). TotalSegmentator: Robust segmentation of 104 anatomical structures in CT images. arXiv.org. https://arxiv.org/abs/2208.05868 . https://doi.org/10.1148/ryai.230024

[3] Jordan, P., Adamson, P. M., Bhattbhatt, V., Beriwal, S., Shen, S., Radermecker, O., Bose, S., Strain, L. S., Offe, M., Fraley, D., Principi, S., Ye, D. H., Wang, A. S., Van Heteren, J., Vo, N.-J., & Schmidt, T. G. (2021). Pediatric Chest/Abdomen/Pelvic CT Exams with Expert Organ Contours (Pediatric-CT-SEG) (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.X0H0-1706

[4] https://www.synapse.org/#!Synapse:syn3193805/wiki/89480

# License
Copyright (c) MONAI Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
