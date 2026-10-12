# MindMap Brain Segmentation

### **Authors**

Sergey Plis and the neuroneural / brainchop-models team (original `model16chan18cls` / `model24chan18cls` training); ported to the MONAI Bundle format by the MONAI Model Zoo community.

### **Tags**

Segmentation, Whole-brain segmentation, Subcortical segmentation, MRI, CT, MeshNet, 3D

## **Model Description**

MindMap segments a 3D medical image into 18 FreeSurfer-style whole-brain classes (gray matter, white matter, ventricles, and subcortical structures such as the thalamus, caudate, putamen, hippocampus and amygdala), and is designed to work across imaging modalities, not just T1 MRI. The network is a MeshNet-style dilated 3D CNN: 13 blocks of `Conv3d(kernel=3, bias=False) -> GroupNorm(num_groups=num_channels, affine=True) -> GELU`, with a symmetric coprime dilation ramp `1, 3, 5, 7, 13, 19, 31, 19, 13, 7, 5, 3, 1` (receptive field 255, matching the 256^3 conform grid with no gridding holes), followed by a `1x1x1` classifier convolution *with* a bias (18 classes). Every voxel is classified independently (dense, whole-volume inference; no patching or sliding window).

This is the `mindmap` model from the brainchop project -- `model16chan18cls`'s exact topology at 24 channels instead of 16 (same 13 hidden layers, same dilation ladder, same 18-class head, same colormap), published publicly as `model24chan18cls_gdice_prio` in [neuroneural/brainchop-test](https://github.com/neuroneural/brainchop-test/tree/webgpu/public/models/model24chan18cls_gdice_prio). This bundle is a port of that model into a standard MONAI bundle, so it can be used with the broader MONAI ecosystem (MONAI Label, MONAI Deploy, and `monai.bundle`'s own export tooling, subject to the checkpoint-loading caveat in **Limitations** below). It does not replace the brainchop project's own native inference engines, which remain the fast path for interactive/production use; this bundle targets MONAI-ecosystem interoperability instead.

This port covers the default 18-class label output only. The upstream reference implementation also has an optional CAT-lite partial-volume-estimation mode, producing gray/white/CSF fraction maps instead of hard labels; that is a separate, more involved feature and is intentionally out of scope for this bundle.

## **Data**

Training data and procedure for the original checkpoint are not documented by this port beyond what brainchop-models states: trained on SynthSeg-style synthetic data (`synth18` / `label18`), converted from a Catalyst `gn_hdc_deep_fast_turbo` checkpoint (macro-Dice ~0.78 per brainchop-models' own README). The reference checkpoint is published, MIT licensed, at <https://github.com/neuroneural/brainchop-models/blob/f53f7b93d725416c2705a9e4f79765d1739badb9/meshnet/model24chan18cls/model.pth> -- the same commit `large_files.yml` pins.

**Checkpoint format note:** despite its `.pth` extension, that file is serialized as safetensors, not a torch pickle/zip archive (brainchop-models' own README calls this out). `scripts/checkpoint.py` loads it with `safetensors.torch.load_file`, not `torch.load`.

`models/model.pt` in this bundle **is that upstream file verbatim** (see `large_files.yml`, pinned to a specific commit with a sha256 `hash_val`), not a locally pre-renamed copy. `scripts/checkpoint.py`'s `load_meshnet_gn_checkpoint` renames its flat `model.<3i>.weight` (conv) / `model.<3i+1>.weight,bias` (GroupNorm) / `model.<N>.weight,bias` (classifier) keys into `MindMapNet`'s `features.<i>.weight` / `classifier.weight,bias` layout at load time (deriving the hidden-block count from the checkpoint itself, and validating there are no unexpected or missing keys, rather than hardcoding it). `configs/inference.json`'s `initialize` step calls this directly instead of the usual ignite `CheckpointLoader`.

#### **Preprocessing**

This bundle reproduces the upstream reference implementation's intensity-clip, conform-reslice, and quantile-normalization preprocessing exactly (see `scripts/conform.py`), not a generic MONAI intensity/spacing transform -- with one exception noted in **Limitations**: input orientation (qform/sform affine selection) uses NiBabel's resolution logic rather than the reference implementation's own:

1. **Intensity clip** -- niimath-style: a 1000-bin histogram over the input's nonzero voxels finds the 98th-percentile intensity, then the image is linearly rescaled into `[0, 255]`.
2. **Conform reslice** -- the image is resampled (trilinear) onto a fixed `256x256x256`, 1mm-isotropic grid in LIA orientation, centered on the input volume's geometric center. Out-of-field voxels are filled with the input's own minimum intensity.
3. **uint8 cast** -- the conformed volume is rounded and clamped to `[0, 255]`.
4. **Quantile normalization** -- a 256-bin histogram gives the 5th and 95th percentile intensities by plain rank selection (mindmap's `quantile_mode: "rank"`: no interpolation, no epsilon added to the denominator, no output clamp -- though a non-positive `high - low` still falls back to a denominator of 1.0, matching the reference implementation's own guard); the volume is rescaled to `(x - low) / (high - low)`. This is what actually reaches the network.

Postprocessing (`scripts/postprocess.py`) mirrors the upstream reference implementation's post-segmentation chain for label models: argmax the 18-channel network output into a label map, keep only the single largest 26-connected component across *all* nonzero classes jointly (voxels outside it are zeroed, voxels inside it keep their predicted class -- this is not a binary mask cleanup), then nearest-neighbor reslice that label map back onto the input's native grid. Passing `save_conform: true` to `MindMapPostprocessd` (wired to the bundle-level `save_conform` config value, `false` by default) skips that last reslice and returns the label map in 256^3 conform space instead, matching the reference implementation's `--save-conform` option.

## **Performance**

The original brainchop-models README reports macro-Dice ~0.78 for the source checkpoint; this bundle does not independently verify that figure.

This port was checked against the upstream reference implementation's own CPU-backend regression fixtures (`t1_crop.nii.gz` -- also published in neuroneural/brainchop-test -- against its conform-space output; a second, 2mm-resolution input against its native-space output): voxel-level label agreement is 98.8% and 97.9% respectively, with roughly 96% of the disagreeing voxels immediately adjacent to a class boundary in the reference (i.e. concentrated at inherently ambiguous tissue-boundary voxels, not a systematic mislabeling of any structure). This is not bit-exact: an 18-class whole-brain segmentation has a large amount of internal class-boundary surface area, so floating-point summation-order differences between this NumPy/PyTorch port and the reference implementation's custom SIMD kernels (different order of the same conv/GroupNorm reductions) flip a small fraction of near-tie voxels. See **Limitations**.

## **Additional Usage Steps**

Run inference with:

```
python -m monai.bundle run \
  --config_file configs/inference.json \
  --meta_file configs/metadata.json \
  --bundle_root . \
  --dataset_dir <folder containing input .nii/.nii.gz files>
```

Output files are written to `<bundle_root>/eval/<case>/<case>_mindmap.nii.gz` by default, as an 8-bit label volume (values 0-17; see `configs/metadata.json`'s `label_classes` for the class names, and the model's published `colormap.json` in neuroneural/brainchop-test for display colors: `[0,0,0]` Unknown, `[245,245,245]` Cerebral-White-Matter, `[205,62,78]` Cerebral-Cortex, `[120,18,134]` Lateral-Ventricle, `[196,58,250]` Inferior-Lateral-Ventricle, `[220,248,164]` Cerebellum-White-Matter, `[230,148,34]` Cerebellum-Cortex, `[0,118,14]` Thalamus, `[122,186,220]` Caudate, `[236,13,176]` Putamen, `[12,48,255]` Pallidum, `[204,182,142]` 3rd-Ventricle, `[42,204,164]` 4th-Ventricle, `[119,159,176]` Brain-Stem, `[220,216,20]` Hippocampus, `[103,255,255]` Amygdala, `[255,165,0]` Accumbens-area, `[165,42,42]` VentralDC). Pass `--save_conform true` to keep the output in 256^3 conform space instead of reslicing to the input's native grid.

This model is designed for **whole-volume** inference: `scripts/network.py`'s `GroupNorm` layers compute their statistics from the entire `256x256x256` conformed volume, so `configs/inference.json` intentionally uses `SimpleInferer`, not `SlidingWindowInferer` -- windowing would compute normalization statistics over a sub-volume instead of the whole brain and would not reproduce the reference implementation's output.

## **System Configuration**

The network itself is small (~195K parameters counting the classifier bias) and runs comfortably on CPU; most of the cost is the `256^3` preprocessing/postprocessing resampling and the largest-connected-component pass, implemented in NumPy/SciPy in this bundle (the upstream reference implementation's own C/CUDA/Metal engines are considerably faster for production/interactive use). No training configuration is provided by this port.

## **Limitations**

- This is a port of a third-party model for MONAI-ecosystem interoperability, not a MONAI Consortium-trained or -validated model. It has not been evaluated by MONAI Model Zoo maintainers for accuracy and is not cleared or approved for clinical or diagnostic use.
- Voxel-level output is not bit-exact against the upstream reference implementation's own CPU backend (see **Performance** above for measured agreement); disagreement is concentrated at inter-class boundary voxels rather than indicating a structural porting error.
- The upstream reference implementation's CAT-lite partial-volume-estimation mode is not implemented in this bundle -- only the default hard-label output is. `--crop`, `--mask`, and `--border` are not applicable to this model (the reference implementation's own model metadata lists all three as unsupported for `mindmap`).
- The upstream reference implementation's Hounsfield-to-Cormack CT conversion and an alternate "comply" preprocessing path are not implemented in this bundle. Both are off by default upstream too, so their absence does not affect the default output.
- Input orientation/qform-sform handling uses NiBabel's affine resolution through MONAI's default NIfTI reader, not the upstream reference implementation's own logic: NiBabel selects the sform when its code is nonzero, otherwise the qform when its code is nonzero, otherwise a fallback affine. The reference implementation instead picks whichever of qform/sform has the numerically higher code, so behavior can diverge on headers where both are coded (or malformed) with a higher-coded qform.
- `models/model.pt` is the upstream checkpoint in its original key layout, not this bundle's `MindMapNet` layout (see **Data** above), so `monai.bundle ckpt_export`/`trt_export` cannot load it directly -- both call ignite's `Checkpoint.load_objects` on the raw file, bypassing this bundle's `scripts.checkpoint.load_meshnet_gn_checkpoint` remapping. `mindmap` is listed in `ci/bundle_custom_data.py`'s `exclude_verify_torchscript_list` for this reason. To export TorchScript, load the weights the normal way first (`monai.bundle run`'s `initialize` step, or `scripts.checkpoint.load_meshnet_gn_checkpoint` directly) and then call `torch.jit.script` on the already-loaded `MindMapNet` instance.

## **License**

This bundle's `LICENSE` file carries two upstream licenses back to back:

- **MIT** (Sergey Plis and the neuroneural / brainchop-models contributors) covers `models/model.pt` (the MIT-licensed `model24chan18cls` checkpoint -- see **Data** above for the runtime key remapping) and the upstream-derived architecture/postprocessing logic in `scripts/network.py` and `scripts/postprocess.py`.
- **BSD-2-Clause** (Chris Rorden's Lab / niimath) covers the niimath-derived intensity-clip and reslice code in `scripts/conform.py` (`_niimath_scale_intensity`, `reslice_to_grid`), ported from niimath by way of the upstream reference implementation's own conform code. `scripts/postprocess.py`'s connected-component filtering is the reference implementation's own independent logic, not niimath-derived, so it falls under the MIT terms above.

## **References**

[1] brainchop-models, the upstream training/checkpoint repository this bundle's weights are pulled from: https://github.com/neuroneural/brainchop-models/tree/f53f7b93d725416c2705a9e4f79765d1739badb9/meshnet/model24chan18cls

[2] neuroneural/brainchop-test, https://github.com/neuroneural/brainchop-test/tree/webgpu/public/models/model24chan18cls_gdice_prio -- publishes this model's weights, config, and reference test input publicly

[3] niimath, https://github.com/rordenlab/niimath -- this bundle's conform/reslice preprocessing (`scripts/conform.py`) ports niimath's algorithms by way of the upstream reference implementation's own conform code; see **License** above.
