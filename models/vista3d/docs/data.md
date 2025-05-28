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
