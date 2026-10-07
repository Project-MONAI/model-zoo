# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Port of brainchopC's default (non-CT, non-comply) preprocessing pipeline
for the mindgrab model: conform.c's `scale_intensity` + `conform_reslice`
(via `brainchopc_conform`), the uint8 cast in brainchopc.c's `run_model`, and
`bc_accel_normalize_input` in backend_accel.c (quantile_mode "linear", the
mode mindgrab's model_meta.json selects).

This reproduces FreeSurfer-style conforming: resample to a 256^3, 1mm
isotropic grid in LIA orientation, centered on the input volume, using
trilinear interpolation. mindgrab's network was trained on volumes
preprocessed this way, so skipping it (e.g. feeding an arbitrary-sized,
arbitrary-orientation volume directly) changes the output.
"""

from __future__ import annotations

import numpy as np

__all__ = ["mindgrab_preprocess", "reslice_to_grid"]


def _niimath_scale_intensity(img: np.ndarray) -> np.ndarray:
    """conform.c `scale_intensity`: niimath's training-intensity transform,
    fixed at f_low=0, f_high=.98. Clips outlier-high intensity using a
    1000-bin histogram over the 98th percentile of nonzero voxels, then
    linearly rescales into [0, 255].
    """
    flat = img.astype(np.float64).ravel()
    n = flat.size
    src_min = float(flat.min())
    src_max = float(flat.max())
    nz = int(np.count_nonzero(np.abs(flat) >= 1e-15))
    nltz = int(np.count_nonzero(flat < 0.0))
    if src_min < 0.0 and 100.0 * nltz / n < 2.0:
        src_min = 0.0

    bin_size = (src_max - src_min) / 1000.0
    if not (bin_size > 0.0):
        return img.astype(np.float32)

    bins = np.clip(((flat - src_min) / bin_size).astype(np.int64), 0, 999)
    hist = np.bincount(bins, minlength=1000)[:1000].astype(np.int64)

    nth = n - int(0.02 * nz)
    cumulative = 0
    idx = 0
    while idx < 999:
        cumulative += int(hist[idx])
        if cumulative + int(hist[idx + 1]) >= nth:
            break
        idx += 1

    clip_max = idx * bin_size + src_min
    scale = 255.0 / (clip_max - src_min) if src_min != clip_max else 1.0
    out = np.clip(0.0 + scale * (flat - src_min), 0.0, 255.0)
    return out.reshape(img.shape).astype(np.float32)


def _conform_affine(in_dim: tuple[int, int, int], in_affine: np.ndarray) -> np.ndarray:
    """conform.c `conform_xform`, `ras=0` branch (brainchopc_conform always
    calls conform_reslice with ras=0): a 256^3, 1mm, LIA-oriented grid,
    centered on the input volume's geometric center.
    """
    half = np.array([in_dim[0] / 2.0, in_dim[1] / 2.0, in_dim[2] / 2.0, 1.0])
    center = (in_affine @ half)[:3]

    out_affine = np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    out_center = (out_affine @ np.array([256.0, 256.0, 256.0, 1.0]))[:3]
    out_affine[0, 3] = center[0] - 0.5 * out_center[0]
    out_affine[1, 3] = center[1] - 0.5 * out_center[1]
    out_affine[2, 3] = center[2] - 0.5 * out_center[2]
    return out_affine


def reslice_to_grid(
    img: np.ndarray,
    in_affine: np.ndarray,
    out_affine: np.ndarray,
    out_shape: tuple[int, int, int],
    linear: bool = True,
    fill: float | None = None,
) -> np.ndarray:
    """conform.c `do_reslice`/`reslice_rows`, generalized over its `linear`
    flag: `linear=True` is the trilinear path used going into conform space
    (mindgrab_preprocess); `linear=False` is the nearest-neighbor path
    `brainchopc_reslice` uses to bring a conform-space prediction back onto
    the input's native grid.

    Out-of-FOV voxels are filled with `img`'s own minimum by default (not
    zero), matching `do_reslice`'s `mn` prefill. In the trilinear case a
    destination voxel is only interpolated if its full 2x2x2 input
    neighborhood is in-bounds -- an all-or-nothing rule, not a partial blend.
    """
    if fill is None:
        fill = float(img.min())
    nx, ny, nz = img.shape
    out_to_in = np.linalg.inv(in_affine) @ out_affine

    # Broadcastable singleton-axis coordinates rather than np.meshgrid's fully
    # materialized (256, 256, 256) arrays: arithmetic broadcasting below
    # produces identical values, but without three redundant ~128MiB float64
    # allocations for a 256^3 grid.
    x = np.arange(out_shape[0], dtype=np.float64)[:, None, None]
    y = np.arange(out_shape[1], dtype=np.float64)[None, :, None]
    z = np.arange(out_shape[2], dtype=np.float64)[None, None, :]

    m = out_to_in
    fxp = m[0, 0] * x + m[0, 1] * y + m[0, 2] * z + m[0, 3]
    fyp = m[1, 0] * x + m[1, 1] * y + m[1, 2] * z + m[1, 3]
    fzp = m[2, 0] * x + m[2, 1] * y + m[2, 2] * z + m[2, 3]

    if not linear:
        ix = np.round(fxp).astype(np.int64)
        iy = np.round(fyp).astype(np.int64)
        iz = np.round(fzp).astype(np.int64)
        valid = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < nx) & (iy < ny) & (iz < nz)
        ix_c = np.clip(ix, 0, nx - 1)
        iy_c = np.clip(iy, 0, ny - 1)
        iz_c = np.clip(iz, 0, nz - 1)
        val = img[ix_c, iy_c, iz_c]
        return np.where(valid, val, fill).astype(np.float32)

    ix = np.floor(fxp).astype(np.int64)
    iy = np.floor(fyp).astype(np.int64)
    iz = np.floor(fzp).astype(np.int64)
    dx = (fxp - ix).astype(np.float32)
    dy = (fyp - iy).astype(np.float32)
    dz = (fzp - iz).astype(np.float32)
    del fxp, fyp, fzp  # not used past this point; drop them before the corner arrays below

    valid = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix + 1 < nx) & (iy + 1 < ny) & (iz + 1 < nz)
    ix_c = np.clip(ix, 0, nx - 2)
    iy_c = np.clip(iy, 0, ny - 2)
    iz_c = np.clip(iz, 0, nz - 2)

    c000 = img[ix_c, iy_c, iz_c]
    c001 = img[ix_c, iy_c, iz_c + 1]
    c010 = img[ix_c, iy_c + 1, iz_c]
    c011 = img[ix_c, iy_c + 1, iz_c + 1]
    c100 = img[ix_c + 1, iy_c, iz_c]
    c101 = img[ix_c + 1, iy_c, iz_c + 1]
    c110 = img[ix_c + 1, iy_c + 1, iz_c]
    c111 = img[ix_c + 1, iy_c + 1, iz_c + 1]

    wx0, wy0, wz0 = 1 - dx, 1 - dy, 1 - dz
    val = (
        c000 * wx0 * wy0 * wz0
        + c001 * wx0 * wy0 * dz
        + c010 * wx0 * dy * wz0
        + c011 * wx0 * dy * dz
        + c100 * dx * wy0 * wz0
        + c101 * dx * wy0 * dz
        + c110 * dx * dy * wz0
        + c111 * dx * dy * dz
    )
    return np.where(valid, val, fill).astype(np.float32)


def _accel_quantile_linear(cumulative: np.ndarray, quantile: float, nvox: int) -> float:
    """backend_accel.c `accel_quantile`, `BC_MODEL_QUANTILE_MODE == 0` branch
    (the "linear" mode mindgrab's model_meta.json selects): a histogram
    approximation of a linearly-interpolated percentile.
    """
    rank = quantile * (nvox - 1)
    base = np.floor(rank)
    fraction = rank - base
    index = int(base)
    low = 0
    while low < 255 and cumulative[low] <= index:
        low += 1
    high = low
    while high < 255 and cumulative[high] <= index + 1:
        high += 1
    if fraction < 0.5:
        return low + (high - low) * fraction
    return high - (high - low) * (1.0 - fraction)


def _quantile_normalize_uint8(
    img_uint8: np.ndarray, q_low: float = 0.02, q_high: float = 0.98, denom_eps: float = 1e-3
) -> np.ndarray:
    """backend_accel.c `bc_accel_normalize_input`, parameterized with
    mindgrab's model_meta.json values (q_low=0.02, q_high=0.98,
    denom_eps=1e-3, clamp=True).
    """
    nvox = img_uint8.size
    hist = np.bincount(img_uint8.ravel().astype(np.int64), minlength=256)[:256]
    cumulative = np.cumsum(hist)
    low = _accel_quantile_linear(cumulative, q_low, nvox)
    high = _accel_quantile_linear(cumulative, q_high, nvox)
    denom = high - low + denom_eps
    if not (denom > 0.0):
        denom = 1.0
    out = (img_uint8.astype(np.float32) - np.float32(low)) / np.float32(denom)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def mindgrab_preprocess(img: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full default (MRI, non-CT, non-comply) brainchopC preprocessing chain
    for mindgrab: scale_intensity -> conform reslice -> uint8 cast ->
    quantile normalize.

    Args:
        img: input volume, shape (nx, ny, nz), any real-valued dtype.
        affine: 4x4 voxel-to-world (RAS mm) affine, i.e. what nibabel/MONAI
            report as the image's affine.

    Returns:
        normalized: float32 array of shape (256, 256, 256), values in [0, 1],
            ready to feed to MindGrabNet.
        conform_affine: the 4x4 voxel-to-world affine of `normalized`'s grid,
            needed to reslice the network's output back onto the input's
            native grid.
    """
    scaled = _niimath_scale_intensity(np.asarray(img))
    conform_affine = _conform_affine(scaled.shape, np.asarray(affine, dtype=np.float64))
    conformed = reslice_to_grid(
        scaled, np.asarray(affine, dtype=np.float64), conform_affine, out_shape=(256, 256, 256), linear=True
    )
    conformed_u8 = np.clip(np.round(conformed), 0, 255).astype(np.uint8)
    normalized = _quantile_normalize_uint8(conformed_u8)
    return normalized, conform_affine
