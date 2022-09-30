
# 3 Label Ventricular Segmentation

This network segments cardiac ventricle in 2D short axis MR images. The left ventricular pool is class 1, left ventricular myocardium class 2, and right ventricular pool class 3. Full cycle segmentation with this network is possible although much of the training data is composed of segmented end-diastole images. The input to the network is single 2D images thus segmenting whole time-dependent volumes consists of multiple inference operations.

The network and training scheme are essentially identical to that described in:

`Kerfoot E., Clough J., Oksuz I., Lee J., King A.P., Schnabel J.A. (2019) Left-Ventricle Quantification Using Residual U-Net. In: Pop M. et al. (eds) Statistical Atlases and Computational Models of the Heart. Atrial Segmentation and LV Quantification Challenges. STACOM 2018. Lecture Notes in Computer Science, vol 11395. Springer, Cham. https://doi.org/10.1007/978-3-030-12029-0_40`

## Data

The dataset used to train this network unfortunately cannot be made public as it contains unreleased image data from King's College London. Existing public datasets such as the[Sunnybrook Cardiac Dataset](http://www.cardiacatlas.org/studies/sunnybrook-cardiac-data/) and [ACDC Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/) set can be used to train a similar network.

The `train.json` configuration assumes all data is stored in a single npz file with keys "images" and "segs" containing respectively the raw image data and their accompanying segmentations. The given network was training with stored volumes with shapes `(9095, 256, 256)` thus other data of differing spatial dimensions must be cropped to `(256, 256)` or zero-padded to that size. For the training data this was done as a preprocessing step but the original pixel values are otherwise unchanged from their original forms.

## Training

The network is trained with this data in conjunction with a series of augmentations for regularisation and robustness. Many of the original images are smaller than the expected size of `(256, 256)` and so were zero-padded, the network can thus be expected to be robust against large amounts of empty space in the inputs. Rotation and zooming is also applied to force the network to learn different sizes and orientations of the heart in the field of view.

Free-form deformation is applied to vary the shape of the heart and its surrounding tissues which mimics to a degree deformation like what would be observed through the cardiac cycle. This of course does not replicate the heart moving through plane during the cycle or represent other observed changes but does provide enough variation that full-cycle segmentation is generally acceptable.

Smooth fields are used to vary contrast and intensity in localised regions to simulate some of the variation in image quality caused by acquisition artefacts. Guassian noise is also added to simulate poor quality acquisition. These together force the network to learn to deal with a wider variation of image quality and partially to account for the difference between scanner vendors.

Training is invoked with the following command line:

```sh
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf --bundle_root .
```

The dataset file is assumed to be `allimages3label.npz` but can be changed by setting the `dataset_file` value to your own file.

## Inference

An example notebook [visualise.ipynb](./visualise.ipynb) demonstrates using the network directly with input images. Inference of 3D volumes only can be accomplished with the `inference.json` configuration:

```sh
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf --dataset_dir dataset --output_dir ./output/ --bundle_root .
```

# License
This model is released under the MIT License. The license file is included with the model.
