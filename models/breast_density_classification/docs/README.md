# Description
A pre-trained model for breast-density classification.

# Model Overview
This model is trained using transfer learning on InceptionV3. The model weights were fine tuned using the Mayo Clinic Data. The details of training and data is outlined in https://arxiv.org/abs/2202.08238. The images should be resampled to a size [299, 299, 3] for training.
A training pipeline will be added to the model zoo in near future.
The bundle does not support torchscript.

# Sample Data
In the folder `sample_data` few example input images are stored for each category of images. These images are stored in jpeg format for sharing purpose.

# Input and Output Formats
The input image should have the size [299, 299, 3]. For a dicom image which are single channel. The channel can be repeated 3 times.
The output is an array with probabilities for each of the four class.

# Commands Example
Create a json file with names of all the input files. Execute the following command
```
python scripts/create_dataset.py -base_dir <path to the bundle root dir>/sample_data -output_file configs/sample_image_data.json
```
Change the `filename` for the field `data` with the absolute path for `sample_image_data.json`


# Add scripts folder to your python path as follows
```
export PYTHONPATH=$PYTHONPATH:<path to the bundle root dir>/scripts
```

# Execute Inference
The inference can be executed as follows
```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json configs/logging.conf
```

# Execute training
It is a work in progress and will be shared in the next version soon.

# Contributors
This model is made available from Center for Augmented Intelligence in Imaging, Mayo Clinic Florida. For questions email Vikash Gupta (gupta.vikash@mayo.edu).

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
