# OIR Flatmount Segmentation (Hartnett Lab)

### **Authors**
Neal S. Shah*, Aniket Ramshekar*, Bright Asare-Bediako, Morgan P. Tankersley, Heng-Chiao Huang, Shreya Beri, Eric Kunz, Aaron Y. Lee, M. Elizabeth Hartnett

Byers Eye Institute Department of Ophthalmology, Stanford University School of Medicine, Stanford, CA, USA

### **Tags**
Segmentation, Retinal Flatmount, Oxygen-Induced Retinopathy, OIR, Mouse, Rat, Intravitreal Neovascularization, Avascular Area

## **Model Description**
This model performs automated segmentation of oxygen-induced retinopathy (OIR) retinal flatmount images into three regions: total retina (TR), intravitreal neovascularization (IVNV), and avascular area (AVA).  
The architecture is a multi-task Attention U-Net with a ConvNeXt-Tiny encoder [1] and deep supervision (~8.7M trainable parameters).  
For inference, the final release uses an ensemble of 5 cross-validation models, with test-time augmentation and per-class thresholding, to improve robustness across mouse and rat OIR images.

## **Data**
Model development used three datasets:

1. **Rat IVNV pretraining dataset:** 72 rat OIR flatmount images with IVNV-only annotations (used in intermediate Stage 2 training).
2. **Final development dataset:** 345 annotated images total (267 mouse, 78 rat), including:
   - 127 expert human-annotated images (49 mouse, 78 rat)
   - 218 curated open-source mouse images [2] with reviewed masks generated from a prior published model [3]
3. **Independent test dataset:** 37 images (18 mouse OIR, 19 rat OIR), held out from training/validation/model selection.

For final model development, a modified 5-fold cross-validation strategy was used, with expert-annotated images serving as fold-level validation references and curated open-source mouse images used in training only.

#### **Preprocessing**
Input retinal flatmount images were converted to grayscale, resized to 512×512, and intensity-normalized.  
During training, joint image-mask augmentation was applied using random horizontal/vertical flips, random rotations (up to 180 degrees), brightness/contrast perturbation, CLAHE, Gaussian noise, elastic/grid/optical distortions, coarse dropout, motion blur, and random gamma adjustments.

## **Performance**

Dice agreement between model masks and human consensus masks was high for total retina (TR) and AVA, and moderate for IVNV in both species:
- Rat: TR Dice=0.983, AVA Dice=0.924, IVNV Dice=0.612
- Mouse: TR Dice=0.975, AVA Dice=0.912, IVNV Dice=0.601

At the metric level, the deep learning model showed strong correlation with the mean of three graders for rat percent AVA (r=0.979) and rat percent IVNV (r=0.943). In mouse OIR, correlation was strong for percent AVA (r=0.957) but weak for percent IVNV (r=0.265), likely due to high inter-grader variability for mouse IVNV scoring.

(For full analysis please refer to the manuscript.)

## **System Configuration**
This model was trained on an Apple M2 pro 16GB Macbook. 5-fold cross-validation was run sequentially with batch size 4 and a maximum of 120 epochs per fold. The folds ran for 86, 88, 120, 61, and 80 epochs, with total training time of approximately 24 hours.

## **Additional Usage Steps** 
Model checkpoints are hosted externally and linked through `large_files.yml` (not committed directly in the repo due to file size limits).  

## **References** 
1. Liu Z, Mao H, Wu CY, Feichtenhofer C, Darrell T, Xie S. A ConvNet for the 2020s. 2022:11966-11976.
2. Marra KV, Chen JS, Robles-Holmes HK, et al. Development of an Open-Source Dataset of Flat-Mounted Images for the Murine Oxygen-Induced Retinopathy Model of Ischemic Retinopathy. Transl Vis Sci Technol. Dec 2 2024;13(12):4.
3. Xiao S, Bucher F, Wu Y, et al. Fully automated, deep learning segmentation of oxygen-induced retinopathy images. JCI Insight. Dec 21 2017;2(24)doi:10.1172/jci.insight.97585 
