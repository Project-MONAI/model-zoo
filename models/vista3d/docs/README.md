# Model Overview
Vista3D model train/inference pipeline

## Training configuration
The training was performed with the following:
- GPU: at least 16GB GPU memory
- Actual Model Input: 128 x 128 x 128
- AMP: True
- Optimizer: Adam
- Learning Rate: 1e-2
- Loss: BCE loss and L1 loss

## Data
Note that VISTA3D is trained from a huge collection of datasets and cannot be simply reproduced in this bundle.

The spleen Task from the Medical Segmentation Decathalon is selected as an example to show how to do train, continuous learning and evaluate. Users can find more details on the datasets at http://medicaldecathlon.com/.

To train with other datasets, users need to provide a json data split for training and continuous learning (`configs/msd_task09_spleen_folds.json` is an example for reference). The data split should meet the following format with a 5-fold split ('testing' labels are optional):
```
{
    "training": [
        {"image": "img0001.nii.gz", "label": "label0001.nii.gz", "fold": 0},
        {"image": "img0002.nii.gz", "label": "label0002.nii.gz", "fold": 2},
        ...
     ],
    "testing": [
        {"image": "img0003.nii.gz", "label": "label0003.nii.gz"},
        {"image": "img0004.nii.gz", "label": "label0004.nii.gz"},
        ...
   ]
}
```

### Input
1 channel
- List of 3D CT patches

### Output
In Training Mode: Training loss

In Evaluation Mode: Segmentation

## Performance

#### TensorRT speedup
The `vista3d` bundle supports acceleration with TensorRT. The table below displays the speedup ratios observed on an A100 80G GPU. Please note for 32bit precision models, they are benchmarked with tf32 weight format.

| method | torch_tf32(ms) | torch_amp(ms) | trt_tf32(ms) | trt_fp16(ms) | speedup amp | speedup tf32 | speedup fp16 | amp vs fp16|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| model computation | 108.53| 91.9 | 106.84 | 60.02 | 1.18 | 1.02 | 1.81 | 1.53 |
| end2end | 6740 | 5166 | 5242 | 3386 | 1.30 | 1.29 | 1.99 | 1.53 |

Where:
- `model computation` means the speedup ratio of model's inference with a random input without preprocessing and postprocessing
- `end2end` means run the bundle end-to-end with the TensorRT based model.
- `torch_tf32` and `torch_amp` are for the PyTorch models with or without `amp` mode.
- `trt_tf32` and `trt_fp16` are for the TensorRT based models converted in corresponding precision.
- `speedup amp`, `speedup tf32` and `speedup fp16` are the speedup ratios of corresponding models versus the PyTorch float32 model
- `amp vs fp16` is the speedup ratio between the PyTorch amp model and the TensorRT float16 based model.

This result is benchmarked under:
 - TensorRT: 10.3.0+cuda12.6
 - Torch-TensorRT Version: 2.4.0
 - CPU Architecture: x86-64
 - OS: ubuntu 20.04
 - Python version:3.10.12
 - CUDA version: 12.6
 - GPU models and configuration: A100 80G

## MONAI Bundle Commands
In addition to the Pythonic APIs, a few command line interfaces (CLI) are provided to interact with the bundle. The CLI supports flexible use cases, such as overriding configs at runtime and predefining arguments in a file.

For more details usage instructions, visit the [MONAI Bundle Configuration Page](https://docs.monai.io/en/latest/config_syntax.html).


#### Execute training:

```
python -m monai.bundle run --config_file configs/train.json
```

Please note that if the default dataset path is not modified with the actual path in the bundle config files, you can also override it by using `--dataset_dir`:

```
python -m monai.bundle run --config_file configs/train.json --dataset_dir <actual dataset path>
```

#### Execute finetune:

```
python -m monai.bundle run --config_file configs/train.json --finetune True --epochs 5
```

Please note that the path of model weights is "/models/model.pt", you can also override it by using `--finetune_model_path`:

```
python -m monai.bundle run --config_file configs/train.json --finetune True --finetune_model_path <actual model weights path>
```

#### Enable early stop in training:

```
python -m monai.bundle run --config_file configs/train.json --early_stop True
```

#### Override the `train` config to execute multi-GPU training:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run --config_file "['configs/train.json','configs/multi_gpu_train.json']"
```


#### Execute continual learning
When finetuning with new class names, please update `configs/train_continual.json`'s `label_mappings` accordingly.

The current label mapping `[[1, 3]]` indicates that training labels' class indices `1` is mapped
to the VISTA model's class `3` (format `[[src_class_0, dst_class_0], [src_class_1, dst_class_1], ...]`). For new classes, user
can map to any value larger than 132.

`label_set` is used to identify the VISTA model classes for providing training prompts.
`val_label_set` is used to identify the original training label classes for computing foreground/background mask during validation.

The default configs for both variables are derived from the `label_mappings` config and include `[0]`:
```
"label_set": "$[0] + list(x[1] for x in @label_mappings#default)"
"val_label_set": "$[0] + list(x[0] for x in @label_mappings#default)"
```
`drop_label_prob` and `drop_point_prob` means percentage to remove class prompts and point prompts respectively. If `drop_point_prob=1`, the
model is only finetuning for automatic segmentation, while `drop_label_prob=1` means only finetuning for interactive segmentation. The VISTA3D foundation
model is trained with interactive only (drop_label_prob=1) and then froze the point branch and trained with fully automatic segmentation (`drop_point_prob=1`).
In this bundle, the training is simplified by jointly training with class prompts and point prompts.

Single-GPU:
```
python -m monai.bundle run \
	--config_file="['configs/train.json','configs/train_continual.json']" --epochs=320 --learning_rate=0.005
```

Multi-GPU:
```
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run \
	--config_file="['configs/train.json','configs/train_continual.json','configs/multi_gpu_train.json']" --epochs=320 --learning_rate=0.005
```

The patch size parameter is defined in `configs/train_continual.json`: `"patch_size": [128, 128, 128]`, and this works for the use cases
of extending the current model to segment a few novel classes. Finetuning all supported classes may require large GPU memory and carefully designed
multi-stage training processes.

Changing `patch_size` to a smaller value such as `"patch_size": [96, 96, 96]` used in `configs/train.json` would reduce the training memory footprint.

In `train_continual.json`, only subset of training and validation data are used, change `n_train_samples` and `n_val_samples` to use full dataset.

In `train.json`, `validate[evaluator][val_head]` can be `auto` and `point`. If `auto`, the validation results will be automatic segmentation. If `point`,
the validation results will be sampling one positive point per object per patch. The validation scheme of combining auto and point is deprecated due to
speed issue.

Note: `valid_remap` is a transform that maps the groundtruth label indexes, e.g. [0,2,3,5,6] to sequential and continuous labels [0,1,2,3,4]. This is
required by monai dice calculation. It is not related to mapping label index to VISTA3D defined global class index. The validation data is not mapped
to the VISTA3D global class index.

#### Execute evaluation
`n_train_samples` and `n_val_samples` are used to specify the number of samples to use for training and validation respectively.

`configs/data.yaml` shows potential configurations for each specific dataset for evaluation.

Single-GPU:
```
python -m monai.bundle run \
	--config_file="['configs/train.json','configs/train_continual.json','configs/evaluate.json','configs/data.yaml']"
```

Multi-GPU:
```
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run \
	--config_file="['configs/train.json','configs/train_continual.json','configs/evaluate.json','configs/mgpu_evaluate.json','configs/data.yaml']"
```


#### Execute inference:
Notice the VISTA3d bundle requires at least one prompt for segmentation. It supports label prompt, which is the index of the class for automatic segmentation.
It also supports point click prompts for binary segmentation. User can provide both prompts at the same time. To segment an image, set the input_dict to
:
```
"input_dict": "$[{'image': '/data/Task09_Spleen/imagesTs/spleen_15.nii.gz', 'label_prompt':[1]}]",
"input_dict": "$[{'image': '/data/Task09_Spleen/imagesTs/spleen_15.nii.gz', 'points':[[138,245,18], [271,343,27]], 'point_labels':[1,0]}]"
```
- The input_dict must contain the absolute path to the nii image file, and must contain at least one prompt. The keys are "label_prompt", "points" and "point_labels".
- label_prompt is in the format of [B], points is [1, N, 3], point_labels is [1, N]. B is number of foreground object. **B must be 1 if label_prompt and points are provided together**
- N is number of click points, 3 is x,y,z coordinates **IN THE ORIGINAL IMAGE SPACE**. The inferer only supports SINGLE OBJECT point click segmentatation.
- point_labels 0 means background, 1 means foreground, -1 means ignoring this point.
- label_prompt and points key can be missing, but cannot be missing at the same time.
- points and point_labels must pe provided together.
- The label_prompt can perform multiple foreground object segmentation, e.g. [2,3,4,5] means segment those classes. Point prompts must NOT be provided.
- For segment everything, use label_prompt: list(set([i+1 for i in range(132)]) - set([22, 23, 15, 25, 19, 2, 26, 27, 28, 29, 117]))
- The point prompts for "Kidney", "Lung", "Bone" (class index [2, 20, 21]) are not allowed since those prompts will be divided into sub-categories (e.g. left kidney and right kidney). Use point prompts for the sub-categories as defined in the inference.json.
```
python -m monai.bundle run --config_file configs/inference.json
```

#### Execute batch inference for segmenting everything
```
python -m monai.bundle run --config_file="['configs/inference.json', 'configs/batch_inference.json']" --input_dir="/data/Task09_Spleen/imagesTr" --output_dir="./eval_task09"
```

`configs/batch_inference.json` by default runs the segment everything workflow (classes defined by `everything_labels`) on all (`*.nii.gz`) files in `input_dir`.
This default is overridable by changing the input folder `input_dir`, or the input image name suffix `input_suffix`, or directly setting the list of filenames `input_list`.

Set `"postprocessing#transforms#0#_disabled_": false` to move the postprocessing to cpu to reduce the GPU memory footprint.

#### Execute inference with the TensorRT model:

```
python -m monai.bundle run --config_file "['configs/inference.json', 'configs/inference_trt.json']"
```


## Automatic segmentation label prompts :
The mapping between organ name and label prompt is in the [json file](labels.json)


## Fast Point Window Inference:
When user click a point, there is no need to perform whole image sliding window inference. Set "use_point_window" to true in the inference.json to enable this function.
A window centered at the clicked points will be used for inference. All values outside of the window will set to be "NaN" unless "prev_mask" is passed to the inferer.
If no point click exists, this function will not be used. Notice if "use_point_window" is true and user provided point clicks, there will be obvious cut-off box artefacts.

# References
- Roth, H., Farag, A., Turkbey, E. B., Lu, L., Liu, J., & Summers, R. M. (2016). Data From Pancreas-CT (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU

- J. Ma et al., "AbdomenCT-1K: Is Abdominal Organ Segmentation a Solved Problem?," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 6695-6714, 1 Oct. 2022, doi: 10.1109/TPAMI.2021.3100536.

- JI YUANFENG. (2022). Amos: A large-scale abdominal multi-organ benchmark for versatile medical image segmentation [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7155725

- Antonelli, M., Reinke, A., Bakas, S. et al. The Medical Segmentation Decathlon. Nat Commun 13, 4128 (2022). https://doi.org/10.1038/s41467-022-30695-9

- Rister, B., Yi, D., Shivakumar, K. et al. CT-ORG, a new dataset for multiple organ segmentation in computed tomography. Sci Data 7, 381 (2020). https://doi.org/10.1038/s41597-020-00715-8

- Jakob Wasserthal. (2022). Dataset with segmentations of 104 important anatomical structures in 1204 CT images (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6802614

- Gibson, E., Giganti, F., Hu, Y., Bonmati, E., Bandula, S., Gurusamy, K., Davidson, B., Pereira, S. P., Clarkson, M. J., & Barratt, D. C. (2018). Multi-organ Abdominal CT Reference Standard Segmentations (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1169361

- Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge https://www.synapse.org/#!Synapse:syn3193805/wiki/217753


# License

## Code License

This project includes code licensed under the Apache License 2.0.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

## Model Weights License

The model weights included in this project are licensed under the NCLS v1 License.

Both licenses' full texts have been combined into a single `LICENSE` file. Please refer to this `LICENSE` file for more details about the terms and conditions of both licenses.
