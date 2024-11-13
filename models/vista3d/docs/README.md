# Model Overview
Vista3D model fintuning/evaluation/inference pipeline. VISTA3D is trained using over 20 partial datasets with more complicated pipeline. To avoid confusion, we will only provide finetuning/continual learning APIs for users to finetune on their
own datasets.

## Continual learning

For continual learning, user can change `configs/train_continual.json`. More advanced users can change configurations in `configs/train.json`. The hyperparameters in `configs/train_continual.json` will overwrite ones in `configs/train.json`. Most hyperparameters are straighforward and user can tell based on their names. We list hyperparameters that needs to be modified.

### Data

The spleen Task from the Medical Segmentation Decathalon is selected as an example to show how to continuous learning. Users can find more details on the datasets at http://medicaldecathlon.com/.

To train with other datasets, users need to provide a json data split for training and continuous learning (`configs/msd_task09_spleen_folds.json` is an example for reference). The data split should meet the following format ('testing' labels are optional):
```json
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

```
Note the data is not the absolute path to the image and label file. The actual image file will be `os.path.join(dataset_dir, data["training"][item]["image"])`, where `dataset_dir` is defined in `configs/train_continual.json`. Also 5-fold cross-validation is not required! `fold=0` is defined in train.json, which means any data item with fold==0 will be used as validation and other fold will be used for training. So if you only have 2 data, you can manually set one data to be validation by setting "fold": 0 in its datalist and the other to be training by setting "fold" to any number other than 0.
```

### Best practice to generate data list
User can use monai to generate the 5-fold data lists. Full exampls can be found in VISTA3D open source [codebase](https://github.com/Project-MONAI/VISTA/blob/main/vista3d/data/make_datalists.py)
```python
from monai.data.utils import partition_dataset
from monai.bundle import ConfigParser
base_url = "/path_to_your_folder/"
json_name = "./your_5_folds.json"
# create matching image and label lists.
# The code to generate the lists is based on your local data structure.
# You can use glob.glob("**.nii.gz") e.t.c.
image_list = ['images/1.nii.gz', 'images/2.nii.gz', ...]
label_list = ['labels/1.nii.gz', 'labels/2.nii.gz', ...]
items = [{"image": img, "label": lab} for img, lab in zip(image_list, label_list)]
# 80% for training 20% for testing.
train_test = partition_dataset(items, ratios=[0.8, 0.2], shuffle=True, seed=0)
print(f"training: {len(train_test[0])}, testing: {len(train_test[1])}")
# num_partitions-fold split for the training set.
train_val = partition_dataset(train_test[0], num_partitions=5, shuffle=True, seed=0)
print(f"training validation folds sizes: {[len(x) for x in train_val]}")
# add the fold index to each training data.
training = []
for f, x in enumerate(train_val):
   for item in x:
      item["fold"] = f
      training.append(item)
# save json file
parser = ConfigParser({})
parser["training"] = training
parser["testing"] = train_test[1]
print(f"writing {json_name}\n\n")
if os.path.exists(json_name):
   logger.warning(f"rewrite existing datalist file: {json_name}")
ConfigParser.export_config_file(parser.config, json_name, indent=4)
```

### Configurations

#### `label_mappings`
The core concept of label_mapping is to convert ground-truth label index of each dataset to a unified class index. For example, "Spleen" in MSD09 groundtruth will be represented by 1, while in AbdomenCT-1K it's 3. We unified a global label index (`docs/labels.json`) to represent all 132 classes, and create a label mapping to map those local index to this global index. So when a user is training on their own dataset, we need to know this mapping.

The current label mapping `[[1, 3]]` indicates that training labels' class indices `1` is mapped
to the VISTA model's class `3` (format `[[src_class_0, dst_class_0], [src_class_1, dst_class_1], ...]`). So during inference, "3" is used to segment spleen.

Since it's finetuning, you can map your local class to any global class. If you use [[1,4]], where "4" represents pancreas, the finetuning can still work but requires more training data and epoch because the class "4" is already assigned and trained with pancreas. If you use [[1,3]], where "3" already represents spleen, the finetuning will converge much faster.

#### Best practice to set label_mapping

For a class that represent the same or similar class as the global index, directly map it to the global index. For example, "mouse left lung" (e.g. index 2 in the mouse dataset) can be mapped to the 28 "left lung upper lobe"(or 29 "left lung lower lobe") with [[2,28]]. After finetuning, 28 now represents "mouse left lung" and will be used for segmentation. If you want to segment 4 substructures of aorta, you can map one of the substructuress to 6 aorta and the rest to any unused classes (class > 132), [[1,6],[2,133],[3,134],[4,135]]. For a completely novel class that none of the VISTA global classes are related, directly map to unused classes (class > 132).
```
NOTE: Do not map to global index value >= 255. `num_classes=255` in the config only represent the maximum mapping index, while the actual output class number only depends on your label_mapping definition. The 255 value in the inference output is also used to represent 'NaN' value.
```
#### `n_train_samples` and `n_val_samples`
In `train_continual.json`, only `n_train_samples` and `n_val_samples` are used for training and validation. Remember to change these two values.

#### `patch_size`
The patch size parameter is defined in `configs/train_continual.json`: `"patch_size": [128, 128, 128]`. For finetuning purposes, this value needs to be changed acccording to user's task and GPU memory. Usually a larger patch_size will give better final results.

#### `resample_to_spacing`
The resample_to_spacing parameter is defined in `configs/train_continual.json` and it represents the resolution the model will be trained on. The `1.5,1.5,1.5` mm default is suitable for large CT organs, but for other tasks, this value should be changed to achive the optimal performance.

#### Advanced user: `drop_label_prob` and `drop_point_prob` (in train.json)
VISTA3D is trained to perform both automatic (class prompts) and interactive point segmentation.
`drop_label_prob` and `drop_point_prob` means percentage to remove class prompts and point prompts during training respectively. If `drop_point_prob=1`, the
model is only finetuning for automatic segmentation, while `drop_label_prob=1` means only finetuning for interactive segmentation. The VISTA3D foundation
model is trained with interactive only (drop_label_prob=1) and then froze the point branch and trained with fully automatic segmentation (`drop_point_prob=1`).
In this bundle, the training is simplified by jointly training with class prompts and point prompts and both of the drop ratio is set to 0.25.
```
NOTE: If user doesn't use interactive segmentation, set `drop_point_prob=1` and `drop_label_prob=0` in train.json might provide a faster and easier finetuning process.
```
#### Other explanatory items
In `train.json`, `validate[evaluator][val_head]` can be `auto` and `point`. If `auto`, the validation results will be automatic segmentation. If `point`,
the validation results will be sampling one positive point per object per patch. The validation scheme of combining auto and point is deprecated due to
speed issue.

In `train_continual.json`, `valid_remap` is a transform that maps the groundtruth label indexes, e.g. [0,2,3,5,6] to sequential and continuous labels [0,1,2,3,4]. This is
required by monai dice calculation. It is not related to mapping label index to VISTA3D defined global class index. The validation data is not mapped
to the VISTA3D global class index.

`label_set` is used to identify the VISTA model classes for providing training prompts.
`val_label_set` is used to identify the original training label classes for computing foreground/background mask during validation.
The default configs for both variables are derived from the `label_mappings` config and include `[0]`:
```
"label_set": "$[0] + list(x[1] for x in @label_mappings#default)"
"val_label_set": "$[0] + list(x[0] for x in @label_mappings#default)"
```

Note: Please ensure the input data header is correct. The output file will use the same header as the input data, but if the input data is missing header information, MONAI will automatically provide some default values for missing values (e.g. `np.eye(4)` will be used if affine information is absent). This may cause a visualization misalignment depending on the visualization tool.


### Commands

Single-GPU:
```bash
python -m monai.bundle run \
	--config_file="['configs/train.json','configs/train_continual.json']" --epochs=320 --learning_rate=0.00005
```

Multi-GPU:
```bash
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run \
	--config_file="['configs/train.json','configs/train_continual.json','configs/multi_gpu_train.json']" --epochs=320 --learning_rate=0.00005
```

### MLFlow support

MLflow can be enabled to track and manage your machine learning experiments. To enable MLflow, set the `use_mlflow` parameter to `True`. Below is an example of how to run a single-GPU training command with MLflow enabled:

```bash
python -m monai.bundle run \
	--config_file="['configs/train.json','configs/train_continual.json']" --epochs=320 --learning_rate=0.00005 --use_mlflow True
```

By default, the data of MLflow is stored in the `mlruns/` folder under the bundle's root directory. To launch the MLflow UI and track your experiment data, follow these steps:

1. Open a terminal and navigate to the root directory of your bundle where the `mlruns/` folder is located.

2. Execute the following command to start the MLflow server. This will make the MLflow UI accessible.

```Bash
mlflow ui
```

## Evaluation
Evaluation can be used to calculate dice scores for the model or a finetuned model. Change the `ckpt_path` to the checkpoint you wish to evaluate. The dice score is calculated on the original image spacing using `invertd`, while the dice score during finetuning is calculated on resampled space.

```
NOTE: Evaluation does not support point evaluation.`"validate#evaluator#hyper_kwargs#val_head` is always set to `auto`.
```

Single-GPU:
```
python -m monai.bundle run \
	--config_file="['configs/train.json','configs/train_continual.json','configs/evaluate.json']"
```

Multi-GPU:
```
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run \
	--config_file="['configs/train.json','configs/train_continual.json','configs/evaluate.json','configs/mgpu_evaluate.json']"
```
#### Other explanatory items
The `label_mapping` in `evaluation.json` does not include `0` because the postprocessing step performs argmax (`VistaPostTransformd`), and a `0` prediction would negatively impact performance. In continuous learning, however, `0` is included for validation because no argmax is performed, and validation is done channel-wise (include_background=False). Additionally, `Relabeld` in `postprocessing` is required to map `label` and `pred` back to sequential indexes like `0, 1, 2, 3, 4` for dice calculation, as they are not in one-hot format. Evaluation does not support `point`, but finetuning does, as it does not perform argmax.

## Inference:
For inference, VISTA3d bundle requires at least one prompt for segmentation. It supports label prompt, which is the index of the class for automatic segmentation.
It also supports point click prompts for binary interactive segmentation. User can provide both prompts at the same time.

All the configurations for inference is stored in inference.json, change those parameters:
### `input_dict`
`input_dict` defines the image to segment and the prompt for segmentation.
```
"input_dict": "$[{'image': '/data/Task09_Spleen/imagesTs/spleen_15.nii.gz', 'label_prompt':[1]}]",
"input_dict": "$[{'image': '/data/Task09_Spleen/imagesTs/spleen_15.nii.gz', 'points':[[138,245,18], [271,343,27]], 'point_labels':[1,0]}]"
```
- The input_dict must include the key `image` which contain the absolute path to the nii image file, and includes prompt keys of `label_prompt`, `points` and `point_labels`.
- The `label_prompt` is a list of length `B`, which can perform `B` foreground objects segmentation, e.g. `[2,3,4,5]`. If `B>1`, Point prompts must NOT be provided.
- The `points` is of shape `[N, 3]` like `[[x1,y1,z1],[x2,y2,z2],...[xN,yN,zN]]`, representing `N` point coordinates **IN THE ORIGINAL IMAGE SPACE** of a single foreground object. `point_labels` is a list of length [N] like [1,1,0,-1,...], which
matches the `points`. 0 means background, 1 means foreground, -1 means ignoring this point. `points` and `point_labels` must pe provided together and match length.
- **B must be 1 if label_prompt and points are provided together**. The inferer only supports SINGLE OBJECT point click segmentatation.
- If no prompt is provided, the model will use `everything_labels` to segment 117 classes:

```Python
list(set([i+1 for i in range(132)]) - set([2,16,18,20,21,23,24,25,26,27,128,129,130,131,132]))
```

- The `points` together with `label_prompts` for "Kidney", "Lung", "Bone" (class index [2, 20, 21]) are not allowed since those prompts will be divided into sub-categories (e.g. left kidney and right kidney). Use `points` for the sub-categories as defined in the `inference.json`.
- To specify a new class for zero-shot segmentation, set the `label_prompt` to a value between 133 and 254. Ensure that `points` and `point_labels` are also provided; otherwise, the inference result will be a tensor of zeros.

### `label_prompt` and `label_dict`
The `label_dict` defined in `docs/labels.json` has in total 132 classes. However, there are 5 we do not support and we keep them due to legacy issue. So in total
VISTA3D support 127 classes.

```
"16, # prostate or uterus" since we already have "prostate" class,
"18, # rectum", insufficient data or dataset excluded.
"130, # liver tumor" already have hepatic tumor.
"129, # kidney mass" insufficient data or dataset excluded.
"131, # vertebrae L6", insufficient data or dataset excluded.
```

These 5 are excluded in the `everything_labels`. Another 7 tumor and vessel classes are also removed since they will overlap with other organs and make the output messy. To segment those 7 classes, we recommend users to directly set `label_prompt` to those indexes and avoid using them in `everything_labels`. For "Kidney", "Lung", "Bone" (class index [2, 20, 21]), VISTA3D did not directly use the class index for segmentation, but instead convert them to their subclass indexes as defined by `subclass` dict. For example, "2-Kidney" is converted to "14-Left Kidney" + "5-Right Kidney" since "2" is defined in `subclasss` dict.

```
Note: if the finetuning mapped the local user data index to global index "2, 20, 21", remove the `subclass` dict from inference.json since those values defined in `subclass` will trigger the wrong subclass segmentation.
```

### `resample_spacing`
The optimal inference resample spacing should be changed according to the task. For monkey data, a high resolution of [1,1,1] showed better automatic inference results. This spacing applies to both automatic and interactive segmentation. For zero-shot interactive segmentation for non-human CTs e.g. mouse CT or even rock/stone CT, using original resolution (set `resample_spacing` to [-1,-1,-1]) may give better interactive results.

### `use_point_window`
When user click a point, there is no need to perform whole image sliding window inference. Set "use_point_window" to true in the inference.json to enable this function.
A window centered at the clicked points will be used for inference. All values outside of the window will set to be "NaN" unless "prev_mask" is passed to the inferer (255 is used to represent NaN).
If no point click exists, this function will not be used. Notice if "use_point_window" is true and user provided point clicks, there will be obvious cut-off box artefacts.

### Inference GPU benchmarks
Benchmarks on a 16GB V100 GPU with 400G system cpu memory.
| Volume size at 1.5x1.5x1.5 mm | 333x333x603 | 512x512x512 | 512x512x768 | 1024x1024x512 | 1024x1024x768 |
| :---: | :---: | :---: | :---: | :---: | :---: |
|RunTime| 1m07s | 2m09s | 3m25s| 9m20s| killed |
## Commands
The bundle only provides single-gpu inference.
### Single image inference
```
python -m monai.bundle run --config_file configs/inference.json
```

### Batch inference for segmenting everything
```
python -m monai.bundle run --config_file="['configs/inference.json', 'configs/batch_inference.json']" --input_dir="/data/Task09_Spleen/imagesTr" --output_dir="./eval_task09"
```

`configs/batch_inference.json` by default runs the segment everything workflow (classes defined by `everything_labels`) on all (`*.nii.gz`) files in `input_dir`.
This default is overridable by changing the input folder `input_dir`, or the input image name suffix `input_suffix`, or directly setting the list of filenames `input_list`.


### Execute inference with the TensorRT model:

```
python -m monai.bundle run --config_file "['configs/inference.json', 'configs/inference_trt.json']"
```

By default, the argument `head_trt_enabled` is set to `false` in `configs/inference_trt.json`. This means that the `class_head` module of the network will not be converted into a TensorRT model. Setting this to `true` may accelerate the process, but there are some limitations:

Since the `label_prompt` will be converted into a tensor and input into the `class_head` module, the batch size of this input tensor will equal the length of the original `label_prompt` list (if no prompt is provided, the length is 117). To make the TensorRT model work on the `class_head` module, you should set a suitable dynamic batch size range. The maximum dynamic batch size can be configured using the argument `max_prompt_size` in `configs/inference_trt.json`. If the length of the `label_prompt` list exceeds `max_prompt_size`, the engine will fall back to using the normal PyTorch model for inference. Setting a larger `max_prompt_size` can cover more input cases but may require more GPU memory (the default value is 4, which requires 16 GB of GPU memory). Therefore, please set it to a reasonable value according to your actual requirements.


### TroubleShoot for Out-of-Memory
- Changing `patch_size` to a smaller value such as `"patch_size": [96, 96, 96]` would reduce the training/inference memory footprint.
- Changing `train_dataset_cache_rate` and `val_dataset_cache_rate` to a smaller value like `0.1` can solve the out-of-cpu memory issue when using huge finetuning dataset.
- Set `"postprocessing#transforms#0#_disabled_": false` to move the postprocessing to cpu to reduce the GPU memory footprint.



### TensorRT speedup
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

# References
- Antonelli, M., Reinke, A., Bakas, S. et al. The Medical Segmentation Decathlon. Nat Commun 13, 4128 (2022). https://doi.org/10.1038/s41467-022-30695-9

- VISTA3D: Versatile Imaging SegmenTation and Annotation model for 3D Computed Tomography. arxiv (2024) https://arxiv.org/abs/2406.05285


# License

## Code License

This project includes code licensed under the Apache License 2.0.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

## Model Weights License

The model weights included in this project are licensed under the NCLS v1 License.

Both licenses' full texts have been combined into a single `LICENSE` file. Please refer to this `LICENSE` file for more details about the terms and conditions of both licenses.
