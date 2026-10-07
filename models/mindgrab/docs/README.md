# MindGrab Skull Stripping

### **Authors**

Armin Fani, Mike Doan, Ian Le, Alex Fedorov, Malte Hoffmann, Chris Rorden, Sergey Plis (original MindGrab model, see **Citation Info**); ported to the MONAI Bundle format by the MONAI Model Zoo community.

### **Tags**

Segmentation, Skull stripping, Brain extraction, MRI, CT, MeshNet, 3D

## **Model Description**

MindGrab removes non-brain tissue from a 3D medical image (skull stripping / brain extraction), and is designed to work across imaging modalities, not just T1 MRI. Per the original paper (see **Citation Info**), its architecture is designed from first principles using a spectral interpretation of dilated convolutions, aiming for a lightweight, fully convolutional model that avoids the deployment/hardware overhead of larger neuroimaging models. Concretely, the network is a MeshNet-style dilated 3D CNN: 25 blocks of `Conv3d(kernel=3, bias=False) -> InstanceNorm3d(no affine) -> GELU`, with dilation cycling `16, 8, 4, 2, 1` five times so the receptive field grows without downsampling the volume, followed by a bias-free `1x1x1` classifier convolution (2 classes: background, brain). Every voxel is classified independently (dense, whole-volume inference; no patching or sliding window), and the instance normalization statistics are computed fresh from the input on every forward pass rather than from a fixed running average.

This bundle is a port of the `mindgrab` model from the brainchopC project -- a C/CUDA/Metal/WASM inference engine for brainchop -- into a standard MONAI bundle, so it can be used with the broader MONAI ecosystem (MONAI Label, MONAI Deploy, and `monai.bundle`'s own export tooling, subject to the checkpoint-loading caveat in **Limitations** below). It does not replace brainchopC's own engine, which remains the fast path for the brainchop app itself; this bundle targets MONAI-ecosystem interoperability instead.

## **Data**

Training data and procedure for the original MindGrab checkpoint are not documented by this port. The reference checkpoint is trained elsewhere (a separate Catalyst/PyTorch training codebase) and published, already renamed to its tinygrad-`MeshNet`-style key convention, at <https://github.com/neuroneural/brainchop-models/blob/49d6de76befd804c6e5a10affbef79349cd8c7f4/meshnet/mindgrab/model.pth> (MIT licensed) -- the same commit `large_files.yml` pins.

`models/model.pt` in this bundle **is that upstream file verbatim** (see `large_files.yml`, pinned to a specific commit with a sha256 `hash_val`), not a locally pre-renamed copy. `scripts/checkpoint.py`'s `load_meshnet_checkpoint` renames its `model.<i>.0.weight` / `model.<N>.weight` keys into `MindGrabNet`'s `features.<i>.weight` / `classifier.weight` layout at load time (deriving the hidden-block count from the checkpoint itself, and validating there are no unexpected keys, rather than hardcoding it) and drops the unused classifier bias. `configs/inference.json`'s `initialize` step calls this directly instead of the usual ignite `CheckpointLoader`.

#### **Preprocessing**

This bundle reproduces brainchopC's full preprocessing pipeline exactly (see `scripts/conform.py`), not a generic MONAI intensity/spacing transform:

1. **Intensity clip** -- niimath-style: a 1000-bin histogram over the input's nonzero voxels finds the 98th-percentile intensity, then the image is linearly rescaled into `[0, 255]`.
2. **Conform reslice** -- the image is resampled (trilinear) onto a fixed `256x256x256`, 1mm-isotropic grid in LIA orientation, centered on the input volume's geometric center. Out-of-field voxels are filled with the input's own minimum intensity.
3. **uint8 cast** -- the conformed volume is rounded and clamped to `[0, 255]`.
4. **Quantile normalization** -- a 256-bin histogram gives the 2nd and 98th percentile intensities (linear interpolation); the volume is rescaled to `(x - low) / (high - low + 1e-3)` and clamped to `[0, 1]`. This is what actually reaches the network.

Postprocessing (`scripts/postprocess.py`) mirrors brainchopC's post-segmentation chain: argmax the 2-channel network output into a binary mask, keep only the largest 26-connected foreground component, nearest-neighbor reslice that mask back onto the input's native grid, and produce the final output as **the original input image with every non-brain voxel floored to the image's own minimum intensity** -- not a separate label volume.

## **Performance**

The original paper (see **Citation Info**) reports a mean Dice score of 95.9 +/- 1.6 across its evaluated datasets and modalities, with up to 40-fold speedups and substantially lower memory use than established skull-stripping methods. This bundle does not independently verify that figure.

## **Additional Usage Steps**

Run inference with:

```
python -m monai.bundle run \
  --config_file configs/inference.json \
  --meta_file configs/metadata.json \
  --bundle_root . \
  --dataset_dir <folder containing input .nii/.nii.gz files>
```

Output files are written to `<bundle_root>/eval/<case>/<case>_mindgrab.nii.gz` by default.

This model is designed for **whole-volume** inference: `scripts/network.py`'s `InstanceNorm3d` layers compute their statistics from the entire `256x256x256` conformed volume, so `configs/inference.json` intentionally uses `SimpleInferer`, not `SlidingWindowInferer` -- windowing would compute normalization statistics over a sub-volume instead of the whole brain and would not reproduce brainchopC's output.

## **System Configuration**

The network itself is small (~146K parameters) and runs comfortably on CPU; most of the cost is the `256^3` preprocessing/postprocessing resampling, implemented in NumPy/SciPy in this bundle (brainchopC's own C/CUDA/Metal engine is considerably faster for production/interactive use). No training configuration is provided by this port.

## **Limitations**

- This is a port of a third-party model for MONAI-ecosystem interoperability, not a MONAI Consortium-trained or -validated model. It has not been evaluated by MONAI Model Zoo maintainers for accuracy and is not cleared or approved for clinical or diagnostic use.
- brainchopC's optional CLI capabilities are not implemented in this bundle: `--ct` (Hounsfield-to-Cormack conversion for CT input), `--comply` (an alternate preprocessing path), `--border` (mask dilation by a physical margin), and `--mask` (writing the binary mask as a separate file, rather than only the skull-stripped image). All of these are off by default in brainchopC too, so their absence does not affect the default output.
- Input orientation/qform-sform handling uses NiBabel's affine resolution through MONAI's default NIfTI reader, not brainchopC's own logic: NiBabel selects the sform when its code is nonzero, otherwise the qform when its code is nonzero, otherwise a fallback affine. brainchopC instead picks whichever of qform/sform has the numerically higher code, so behavior can diverge on headers where both are coded (or malformed) with a higher-coded qform.
- `models/model.pt` is the upstream checkpoint in its original key layout, not this bundle's `MindGrabNet` layout (see **Data** above), so `monai.bundle ckpt_export`/`trt_export` cannot load it directly -- both call ignite's `Checkpoint.load_objects` on the raw file, bypassing this bundle's `scripts.checkpoint.load_meshnet_checkpoint` remapping. `mindgrab` is listed in `ci/bundle_custom_data.py`'s `exclude_verify_torchscript_list` for this reason. To export TorchScript, load the weights the normal way first (`monai.bundle run`'s `initialize` step, or `scripts.checkpoint.load_meshnet_checkpoint` directly) and then call `torch.jit.script` on the already-loaded `MindGrabNet` instance.

## **License**

This bundle's `LICENSE` file carries two upstream licenses back to back, matching brainchopC's own `THIRD_PARTY_NOTICES.md`:

- **MIT** (`spikedoanz`; `spikedoanz, Sergey Plis`) covers `models/model.pt` (the verbatim MIT-licensed MindGrab checkpoint -- see **Data** above for the runtime key remapping) and the brainchopC-derived architecture/postprocessing logic in `scripts/network.py` and `scripts/postprocess.py`.
- **BSD-2-Clause** (Chris Rorden's Lab / niimath) covers the niimath-derived intensity-clip and reslice-to-native-grid code in `scripts/conform.py` (`_niimath_scale_intensity`, and the reslice path `scripts/postprocess.py` reuses from it), ported from niimath by way of brainchopC's `src/conform.c`. `scripts/postprocess.py`'s connected-component filtering is brainchopC's own independent implementation, not niimath-derived, so it falls under the MIT terms above.

## **Citation Info**

If you use this model, please cite the original MindGrab paper:

```
@article{fani2026mindgrab,
  title   = {MindGrab: A spectrally-motivated architecture for accessible deep learning in neuroimaging},
  author  = {Fani, Armin and Doan, Mike and Le, Ian and Fedorov, Alex and Hoffmann, Malte and Rorden, Chris and Plis, Sergey},
  journal = {NeuroImage},
  volume  = {338},
  pages   = {122074},
  year    = {2026},
  doi     = {10.1016/j.neuroimage.2026.122074}
}
```

## **References**

[1] Fani, A., Doan, M., Le, I., Fedorov, A., Hoffmann, M., Rorden, C., Plis, S. "MindGrab: A spectrally-motivated architecture for accessible deep learning in neuroimaging." NeuroImage 338 (2026): 122074. https://doi.org/10.1016/j.neuroimage.2026.122074

[2] brainchopC, the C reference implementation this bundle ports (private repository)

[3] MindGrab reference checkpoint, https://github.com/neuroneural/brainchop-models/blob/49d6de76befd804c6e5a10affbef79349cd8c7f4/meshnet/mindgrab/model.pth

[4] niimath, https://github.com/rordenlab/niimath -- this bundle's conform/reslice preprocessing (`scripts/conform.py`) ports niimath's algorithms by way of brainchopC's `src/conform.c`; see **License** above.
