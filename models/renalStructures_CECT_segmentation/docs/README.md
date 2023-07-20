# Model Title
Renal structures CECT segmentation

### **Authors**
Ivan Chernenkiy,   Michael Chernenkiy,   Dmitry Fiev,   Evgeny Sirota, Center for Neural Network Technologies / Institute of Urology and Human Reproductive Systems / Sechenov First Moscow State Medical University

### **Tags**
Segmentation, CT, CECT, Kidney, Renal, Supervised

## **Model Description**
The model is the SegResNet architecture[1] for volumetric (3D) renal structures segmentation. Input is artery, vein, excretory phases after mutual registration and concatenated to 3 channel 3D tensor.


## **Data**
DICOM data from 41 patients with kidney neoplasms were used [2]. The images and segmentation data are available under a CC BY-NC-SA 4.0 license. Data included all phases of contrast-enhanced multispiral computed tomography. We split the data: 32 observations for the training set and 9 – for the validation set. At the labeling stage, the arterial, venous, and excretory phases were taken, affine registration was performed to jointly match the location of the kidneys, and noise was removed using a median filter and a non-local means filter. Validation set ip published to Yandex.Disk. You can download via [link](https://disk.yandex.ru/d/pWEKt6D3qi3-aw) or use following command:
```bash
python -m monai.bundle run download_data --meta_file configs/metadata.json --config_file "['configs/train.json', 'configs/evaluate.json']"
```

**NB**: underlying data is in LPS orientation. IF! you want to test model on your own data, reorient it from RAS to LPS with `Orientation` transform. You can see example of preprocessing pipeline in `inference.json` file of this bundle.

#### **Preprocessing**
Images are (1) croped to kidney region, all (artery,vein,excret) phases are (2) [registered](https://simpleitk.readthedocs.io/en/master/registrationOverview.html#lbl-registration-overview) with affine transform, noise removed with (3) median and (4) non-local means filter. After that, images are (5) resampled to (0.8,0.8,0.8) density and intesities are (6) scaled from [-1000,1000] to [0,1] range.

## **Performance**
On the validation subset, the values of the Dice score of the SegResNet architecture were: 0.89 for the normal parenchyma of the kidney, 0.58 for the kidney neoplasms, 0.86 for arteries, 0.80 for veins, 0.80 for ureters.

When compared with the nnU-Net model, which was trained on KiTS 21 dataset, the Dice score was greater for the kidney parenchyma in SegResNet – 0.89 compared to three model variants: lowres – 0.69, fullres – 0.70, cascade – 0.69. At the same time, for the neoplasms of the parenchyma of the kidney, the Dice score was comparable: for SegResNet – 0.58, for nnU-Net fullres – 0.59; lowres and cascade had lower Dice score of 0.37 and 0.45, respectively. To reproduce, visit - https://github.com/blacky-i/nephro-segmentation


## **Additional Usage Steps**

#### Execute training:

```bash
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json
```
Expected result: finished, Training process started


#### Execute training with finetuning
```bash
python -m monai.bundle run training --dont_finetune false --meta_file configs/metadata.json --config_file configs/train.json
```
Expected result: finished, Training process started, model variables are restored

#### Execute validation:

Download validation data (described in [Data](#data) section).

With provided model weights mean dice score is expected to be ~0.78446.

#####  Run validation script:
```bash
python -m monai.bundle run evaluate --meta_file configs/metadata.json --config_file "['configs/train.json', 'configs/evaluate.json']"
```
Expected result: finished, `Key metric: val_mean_dice best value: ...` is printed.

## **System Configuration**
The model was trained for 10000 epochs on 2 RTX2080Ti GPUs with [SmartCacheDataset](https://docs.monai.io/en/stable/data.html#smartcachedataset). This takes 1 days and 2 hours, with 4 images per GPU.
Training progress is available on [tensorboard.dev](https://tensorboard.dev/experiment/VlEMjLdURH6SyFp216dFBg)

To perform training in minimal settings, at least one 12GB-memory GPU is required.
Actual Model Input: 96 x 96 x 96

## **Limitations**
For developmental purposes only and cannot be used directly for clinical procedures.

## **Citation Info**
```
@article{chernenkiy2023segmentation,
  title={Segmentation of renal structures based on contrast computed tomography scans using a convolutional neural network},
  author={Chernenkiy, IМ and Chernenkiy, MM and Fiev, DN and Sirota, ES},
  journal={Sechenov Medical Journal},
  volume={14},
  number={1},
  pages={39--49},
  year={2023}
}
```

## **References**

[1] Myronenko, A. (2019). 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization. In: Crimi, A., Bakas, S., Kuijf, H., Keyvan, F., Reyes, M., van Walsum, T. (eds) Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. BrainLes 2018. Lecture Notes in Computer Science(), vol 11384. Springer, Cham. https://doi.org/10.1007/978-3-030-11726-9_28

[2] Chernenkiy, I. М., et al. "Segmentation of renal structures based on contrast computed tomography scans using a convolutional neural network." Sechenov Medical Journal 14.1 (2023): 39-49.https://doi.org/10.47093/2218-7332.2023.14.1.39-49

#### **Tests used for bundle checking**

Checking with ci script file
```bash
python ci/verify_bundle.py -b renalStructures_CECT_segmentation -p models
```
Expected result: passed, model.pt file downloaded


Checking downloading validation data file
```bash
cd models/renalStructures_CECT_segmentation
python -m monai.bundle run download_data --meta_file configs/metadata.json --config_file "['configs/train.json', 'configs/evaluate.json']"
```
Expected result: finished, `data/` folder is created and filled with images.


Checking evaluation script
```bash
python -m monai.bundle run evaluate --meta_file configs/metadata.json --config_file "['configs/train.json', 'configs/evaluate.json']"
```
Expected result: finished, `Key metric: val_mean_dice best value: ...` is printed.


Checking train script
```bash
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json
```
Expected result: finished, Training process started


Checking train script with finetuning
```bash
python -m monai.bundle run training --dont_finetune false --meta_file configs/metadata.json --config_file configs/train.json
```
Expected result: finished, Training process started, model variables are restored

Checking inference script
```bash
python -m monai.bundle run inference --meta_file configs/metadata.json --config_file configs/inference.json
```
Expected result: finished, in `eval` folder masks are created

Check unit test with script:
```bash
python ci/unit_tests/runner.py --b renalStructures_CECT_segmentation
```
