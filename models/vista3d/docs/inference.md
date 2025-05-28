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
The `label_dict` defined in `configs/metadata.json` has in total 132 classes. However, there are 5 we do not support and we keep them due to legacy issue. So in total
VISTA3D support 127 classes.

```
"16, # prostate or uterus" since we already have "prostate" class,
"18, # rectum", insufficient data or dataset excluded.
"130, # liver tumor" already have hepatic tumor.
"129, # kidney mass" insufficient data or dataset excluded.
"131, # vertebrae L6", insufficient data or dataset excluded.
```

These 5 are excluded in the `everything_labels`. Another 7 tumor and vessel classes are also removed since they will overlap with other organs and make the output messy. To segment those 7 classes, we recommend users to directly set `label_prompt` to those indexes and avoid using them in `everything_labels`. For "Kidney", "Lung", "Bone" (class index [2, 20, 21]), VISTA3D did not directly use the class index for segmentation, but instead convert them to their subclass indexes as defined by `subclass` dict. For example, "2-Kidney" is converted to "14-Left Kidney" + "5-Right Kidney" since "2" is defined in `subclasss` dict.

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



### Execute inference with the TensorRT model:

```
python -m monai.bundle run --config_file "['configs/inference.json', 'configs/inference_trt.json']"
```

By default, the argument `head_trt_enabled` is set to `false` in `configs/inference_trt.json`. This means that the `class_head` module of the network will not be converted into a TensorRT model. Setting this to `true` may accelerate the process, but there are some limitations:

Since the `label_prompt` will be converted into a tensor and input into the `class_head` module, the batch size of this input tensor will equal the length of the original `label_prompt` list (if no prompt is provided, the length is 117). To make the TensorRT model work on the `class_head` module, you should set a suitable dynamic batch size range. The maximum dynamic batch size can be configured using the argument `max_prompt_size` in `configs/inference_trt.json`. If the length of the `label_prompt` list exceeds `max_prompt_size`, the engine will fall back to using the normal PyTorch model for inference. Setting a larger `max_prompt_size` can cover more input cases but may require more GPU memory (the default value is 4, which requires 16 GB of GPU memory). Therefore, please set it to a reasonable value according to your actual requirements.


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
