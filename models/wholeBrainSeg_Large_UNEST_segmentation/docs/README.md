# Description
A pre-trained model for inferencing whole brain segmentation with 133 structures.
A tutorial and release of model for whole brain segmentation. 

Authors: 
Xin Yu (xin.yu@vanderbilt.edu) (Primary)


Yinchi Zhou (yinchi.zhou@vanderbilt.edu) | Yucheng Tang (yuchengt@nvidia.com) 


# Model Overview
A pre-trained larger UNEST base model [1] for volumetric (3D) whole brain segmentation with T1w MR images.

## Data
The training data is from the Vanderbilt University and Vanderbilt University Medical Center with public released OASIS and CANDI datsets.




The data and segmentation demonstration is as follow:



## Method and Network

The UNEST model is a 3D hierarchical transformer-based semgnetation network.

Details of the architecture:
![](./wholebrain.png) <br>

## Training configuration
The training was performed with at least one 16GB-memory GPU.

Actual Model Input: 96 x 96 x 96

## Input and output formats
Input: 1 channel CT image


## Performance


## commands example
Download trained checkpoint model to ./model/model.pt:


Add scripts component:  To run the workflow with customized components, PYTHONPATH should be revised to include the path to the customized component:

```
export PYTHONPATH=$PYTHONPATH:"'<path to the bundle root dir>/scripts'"

```


Execute inference:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf
```


## More examples output



# Disclaimer
This is an example, not to be used for diagnostic purposes.

# References
[1] Yu, Xin, Yinchi Zhou, Yucheng Tang et al. "Characterizing Renal Structures with 3D Block Aggregate Transformers." arXiv preprint arXiv:2203.02430 (2022). https://arxiv.org/pdf/2203.02430.pdf

[2] Zizhao Zhang et al. "Nested Hierarchical Transformer: Towards Accurate, Data-Efficient and Interpretable Visual Understanding." AAAI Conference on Artificial Intelligence (AAAI) 2022
