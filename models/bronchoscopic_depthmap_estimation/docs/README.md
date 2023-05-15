# Model Title

### **Authors**
*Anyone who should be attributed as part of the model. If multiple people or companies, use a comma seperated list*

Example:

Firstname1 LastName1, Firstname2 Lastname2, Affiliation1

### **Tags**
*What tags describe the model and task performed? Use a comma seperated list*

Example:

Segmentation, MR, Spleen, Self-Supervised

## **Model Description**
*This section should describe general information about the model and task that it performs.  Any high-level reference of what the model is doing, general architecture of the model, and brief mention of data used.  A developer should be able to understand the purpose of the model from this section alone.*

Example:
This model is trained using the UNet architecture [1] and is used for volumetric 3D segmentation of the spleen from CT image. The segmentation of spleen region is formulated as the voxel-wise binary classification. Each voxel is predicted as either foreground (spleen) or background. And the model is optimized with gradient descent method minimizing soft dice loss between the predicted mask and ground truth segmentation.


## **Data**
*This section should talk about the data. Where is the dataset from? How many images? What is the split for training, testing, and validation? Can the users get the data and where? Also, if you had to prepare the data in any particular way to make it work with the model, what steps or preprocessing steps (not transform steps) did you need to take to make it work.*

Example:

The DETR model was trained on COCO 2017 panoptic, a dataset consisting of 118k/5k annotated images for training/validation respectively.

#### **Preprocessing**
Images are resized/rescaled such that the shortest side is at least 800 pixels and the largest side at most 1333 pixels, and normalized across the RGB channels with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).

#### **Additional section**
Any additional sections can be added with h4 (####) and bolded header to make sure it's easily navigable.


## **Performance**
*What sort of training or evaluation performance should people expect from this model? How long did it take you to train the model? If you have training/evaluation charts, you can include them in this section, but aren't required.*

Example:

This model achieves the following results on COCO 2017 validation: a box AP (average precision) of 38.8, a segmentation AP (average precision) of 31.1 and a PQ (panoptic quality) of 43.4.

For more details regarding evaluation results, we refer to table 5 of the original paper.


## **Citation Info**

Example:

```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```
