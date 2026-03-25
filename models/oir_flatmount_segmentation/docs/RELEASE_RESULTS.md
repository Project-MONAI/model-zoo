# Release Results (v2_20260317)

## Finalized Method

- Selected production method: **5-fold ensemble** using each fold's `best.pth` checkpoint and calibrated `thresholds.json`.
- Rationale: highest robustness and expected generalization versus single-fold deployment.

## Cross-Validation Summary

- Fold 0 best validation Dice: `0.8030`
- Fold 1 best validation Dice: `0.8307`
- Fold 2 best validation Dice: `0.8255`
- Fold 3 best validation Dice: `0.8520`
- Fold 4 best validation Dice: `0.8156`

Aggregate:

- Mean best validation Dice: `0.8254`
- Standard deviation: `0.0183`

Reference files:

- `/Users/nealshah/Documents/OIR_segmentation_images/output_kfold_v2/cv_summary.csv`
- `/Users/nealshah/Documents/OIR_segmentation_images/output_kfold_v2/cv_summary.json`

## Thresholds Per Fold

- Fold 0: retina `0.30`, NV `0.95`, VO `0.85`
- Fold 1: retina `0.80`, NV `0.60`, VO `0.65`
- Fold 2: retina `0.75`, NV `0.65`, VO `0.85`
- Fold 3: retina `0.35`, NV `0.95`, VO `0.50`
- Fold 4: retina `0.90`, NV `0.75`, VO `0.40`

## External Validation Outputs

Validation inference outputs are intentionally excluded from this deployable package.
They can be generated later as needed and reviewed before any public release.

## Release Package

Primary release artifact directory:

- `/Users/nealshah/Documents/OIR_segmentation_images/oir_flatmount_segmentation_release_v2_20260317`

Contains:

- `weights/fold_*/model.pth`
- `weights/fold_*/thresholds.json`
- `cv_summary.csv`, `cv_summary.json`
- `release_manifest.json` (includes SHA256 per fold checkpoint)
- `cv_learning_curves.png`

