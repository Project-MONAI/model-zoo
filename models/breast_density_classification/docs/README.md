# Description
A pre-trained model for breast-density classification.

# Model Overview
This model is trained using transfer learning on InceptionV3. The model weights were fine tuned using the Mayo Clinic Data. The details of training and data is outlined in https://arxiv.org/abs/2202.08238.

The bundle does not support torchscript.
# Input and Output Formats
The input image should have the size [299, 299, 3]. For a dicom image which are single channel. The channel can be repeated 3 times. 
The output is an array with probabilities for each of the four class. 

# Sample Data
In the folder `sample_data` few example input images are stored for each category of images. These images are stored in jpeg format for sharing purpose. 

# Commands Example
Create a json file with names of all the input files. Execute the following command
```
python scripts/create_dataset.py config/sample_image_data.json
```
# Add scripts folder to your python path as follows
```
export PATH=${PATH}:<absolute path to your script folder>
```

# Execute Inference 
The inference can be executed as follows 
```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json configs/logging.conf
```

# Execute training
It is a work in progress and will be shared soon.

# Contributors
This model is made available from Center for Augmented Intelligence in Imaging, Mayo Clinic Florida. For questions email Vikash Gupta (gupta.vikash@mayo.edu).
