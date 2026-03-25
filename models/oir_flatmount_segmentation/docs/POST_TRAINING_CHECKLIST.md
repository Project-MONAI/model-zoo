# Post-Training Checklist (v2_20260317)

## Completed

- 5-fold CV retraining finished successfully with joint TR/IVNV/AVA optimization.
- Finalized production method: 5-fold ensemble of `best.pth` with per-fold calibrated thresholds.
- Per-fold outputs confirmed:
  - `best.pth`, `final.pth`, `thresholds.json`
  - `training_history.csv`, `learning_curves.png`
  - `dataset_log.xlsx`, `run_manifest.json`
- Cross-fold summary generated:
  - `output_kfold_v2/cv_summary.csv`
  - `output_kfold_v2/cv_summary.json`
- Release package created:
  - `/Users/nealshah/Documents/OIR_segmentation_images/oir_flatmount_segmentation_release_v2_20260317`
  - includes copied fold weights, thresholds, CV summary, hashes (`release_manifest.json`)
- MONAI bundle local check:
  - `verify_net_in_out` passed for `configs/inference.json`
- Publication repository handoff complete:
  - copied fold weights and thresholds to `/Users/nealshah/Documents/OIR model/weights/fold_*`
- Default inference ensemble path switched to v2 in `infer.py`.

## Pending (for external submission)

- Upload fold model files to public artifact hosting and replace placeholder URLs in `large_files.yml`.
- Decide and add final dataset license/legal wording in `docs/data_license.txt`.
- Run full MONAI metadata validation from the MONAI model-zoo repository context (the schema URL currently referenced by metadata in this environment resolves to 404, so local `verify_metadata` cannot complete against official schema here).
- Optional: add TorchScript export artifact and config if required by deployment target.

## Key Aggregate CV Result

- Mean best validation Dice (across folds): `0.8254`
- Standard deviation: `0.0183`

