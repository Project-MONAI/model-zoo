### Configurations



#### Best practice to set label_mapping

For a class that represent the same or similar class as the global index, directly map it to the global index. For example, "mouse left lung" (e.g. index 2 in the mouse dataset) can be mapped to the 28 "left lung upper lobe"(or 29 "left lung lower lobe") with [[2,28]]. After finetuning, 28 now represents "mouse left lung" and will be used for segmentation. If you want to segment 4 substructures of aorta, you can map one of the substructuress to 6 aorta and the rest to any value, [[1,6],[2,133],[3,134],[4,135]].
```
NOTE: Do not map to global index value >= 255. `num_classes=255` in the config only represent the maximum mapping index, while the actual output class number only depends on your label_mapping definition. The 255 value in the inference output is also used to represent 'NaN' value.
```

#### `val_at_start`
Default `true`, VISTA3D will perform out-of-the-box segmentation before the training.
Users can disable if the validation takes too long.


#### `n_train_samples` and `n_val_samples`
In `train_continual.json`, only `n_train_samples` and `n_val_samples` are used for training and validation.

#### `patch_size`
The patch size parameter is defined in `configs/train_continual.json`: `"patch_size": [128, 128, 128]`. For finetuning purposes, this value needs to be changed acccording to user's task and GPU memory. Usually a larger patch_size will give better final results. `[192,192,128]` is a good value for larger memory GPU.

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
