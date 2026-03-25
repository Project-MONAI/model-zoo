# OIR Flatmount Segmentation (Hartnett Lab)

## Overview

This bundle provides automated segmentation of oxygen-induced retinopathy (OIR) flatmount images for:

- Total Retina (TR)
- Intravitreal Neovascularization (IVNV)
- Avascular Area (AVA)

The model is a multi-task Attention U-Net with a ConvNeXt-Tiny encoder and deep supervision, trained with fold-wise threshold calibration and ensemble inference.

## Intended Use

- Research use in preclinical retinal OIR image analysis.
- Automated quantification support for TR, IVNV, and AVA area measurements.
- Batch processing workflows for publication and reproducible analytics.

## Not Intended For

- Clinical diagnosis, triage, or autonomous treatment decisions.
- Use outside domain conditions (non-flatmount, unrelated species, unrelated disease context) without additional validation.

## Input and Output

- **Input**: Single-channel retinal flatmount image. RGB inputs are converted to grayscale.
- **Output**:
  - Binary masks for TR, IVNV, AVA
  - Overlay visualizations
  - Per-image quantitative metrics (areas and percentages)

## Training Summary

- Data split strategy:
  - Deduplicated by basename to avoid `.jpg/.tif` leakage.
  - HQ expert annotations reserved for validation folds.
  - Auto-generated images included only in training.
- Loss:
  - BCE + Dice (all channels)
  - Focal Tversky (TR/IVNV/AVA channel-weighted)
  - Boundary Dice with Sobel gradients
  - Deep supervision loss at decoder intermediates
- Optimization:
  - AdamW + warmup cosine LR
  - Discriminative learning rates (backbone vs decoder/heads)
  - EMA weights
  - Gradient clipping
  - Early stopping
  - Strong augmentation

## Inference Summary

- Ensemble over fold checkpoints (`fold_*/best.pth`)
- D4 test-time augmentation (rotations and flips)
- Threshold calibration and post-processing:
  - mask binarization by per-class threshold
  - retina-constrained IVNV/AVA
  - optional component filtering and morphological closing

## Reproducibility Artifacts

Each training fold is expected to emit:

- `training_history.csv`
- `learning_curves.png`
- `dataset_log.xlsx`
- `run_manifest.json`
- `best.pth`, `final.pth`, `thresholds.json`

## Authors

Neal Shah1*, Aniket Ramshekar1*, Bright Asare-Bediako1, Morgan Tankersley1, Heng-Chiao Huang1,2, Shreya Beri1, Eric Kunz3, Aaron Y. Lee4, M. Elizabeth Hartnett1,#

1 Byers Eye Institute Department of Ophthalmology, Stanford University School of Medicine, Stanford, California, USA  
2 Department of Ophthalmology, Chang Gung Memorial Hospital, Chiayi, Taiwan  
3 John A. Moran Eye Center, University of Utah, Salt Lake City, Utah, USA  
4 John F. Hardesty Department of Ophthalmology and Visual Sciences, Washington University in St. Louis, St. Louis, Missouri, USA

## Contacts

- Neal Shah: neals1@stanford.edu
- Aniket Ramshekar: aniket.ramshekar@stanford.edu
- M. Elizabeth Hartnett: me.hartnett@stanford.edu

## Citation

If you use this model, please cite the associated TVST publication (to be updated after acceptance) and acknowledge the Hartnett Lab.

## Known Limitations

- Performance may degrade for out-of-distribution scanners/prep protocols.
- Small IVNV lesions are sensitive to threshold and component filtering settings.
- Cross-species domain shift (mouse vs rat) should be evaluated explicitly per cohort.

