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

"""Loads MindMapNet weights directly from the upstream brainchop-models
checkpoint (https://github.com/neuroneural/brainchop-models,
meshnet/model24chan18cls/model.pth), instead of shipping a pre-renamed copy.

That checkpoint is a flat, positionally-named PyTorch `nn.Sequential`-style
state dict (see `tools/convert_catalyst_gn_hdc_deep.py` in brainchop-models):
three entries per hidden block -- `model.<3i>.weight` (the block's Conv3d,
bias-free), `model.<3i+1>.weight` / `model.<3i+1>.bias` (the block's
GroupNorm, *with* affine; the GELU activation contributes no entry) --
followed by a final `model.<N>.weight` / `model.<N>.bias` for the classifier
conv, which does carry a bias.

Despite its `.pth` extension, the file is serialized as safetensors, not a
torch pickle/zip archive (brainchop-models' own README calls this out: "fp32
weights (safetensors; brainchop `safe_load`s it)") -- `torch.load` cannot
parse it, so this module loads it with `safetensors.torch.load_file` instead.
"""

from __future__ import annotations

import hashlib
import re

import torch
from safetensors.torch import load_file as load_safetensors

__all__ = ["convert_meshnet_gn_checkpoint", "load_meshnet_gn_checkpoint"]

_WEIGHT_KEY_RE = re.compile(r"^model\.(\d+)\.weight$")


def convert_meshnet_gn_checkpoint(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a brainchop-models flat-Sequential MeshNet-with-affine-GroupNorm
    state dict into MindMapNet's layout. The hidden-block count is derived
    from the checkpoint itself (from the highest `model.<idx>.weight` index,
    which is always the classifier, three slots per hidden block), not
    hardcoded, so this works for any checkpoint sharing the same convention
    (e.g. model16chan18cls's identically-shaped checkpoint).

    Raises ValueError if the checkpoint doesn't match that convention exactly
    (a non-multiple-of-3 layout, missing GroupNorm/classifier params, or extra
    keys), rather than silently loading something wrong.
    """
    weight_indices = {int(m.group(1)) for key in state_dict if (m := _WEIGHT_KEY_RE.match(key))}
    if not weight_indices:
        raise ValueError("no 'model.<idx>.weight' keys found -- not a flat-Sequential MeshNet checkpoint")

    classifier_idx = max(weight_indices)
    if classifier_idx % 3 != 0 or classifier_idx == 0:
        raise ValueError(f"expected the classifier at an index that is a positive multiple of 3, got {classifier_idx}")
    n_hidden = classifier_idx // 3

    expected_conv_indices = {3 * i for i in range(n_hidden)}
    expected_gn_indices = {3 * i + 1 for i in range(n_hidden)}
    if weight_indices != expected_conv_indices | expected_gn_indices | {classifier_idx}:
        raise ValueError(
            f"unexpected 'model.<idx>.weight' indices: got {sorted(weight_indices)}, "
            f"expected conv indices {sorted(expected_conv_indices)}, GroupNorm indices "
            f"{sorted(expected_gn_indices)}, and classifier index {classifier_idx}"
        )

    expected_keys = (
        {f"model.{i}.weight" for i in expected_conv_indices}
        | {f"model.{i}.weight" for i in expected_gn_indices}
        | {f"model.{i}.bias" for i in expected_gn_indices}
        | {f"model.{classifier_idx}.weight", f"model.{classifier_idx}.bias"}
    )
    unexpected = set(state_dict) - expected_keys
    if unexpected:
        raise ValueError(f"unexpected keys in checkpoint: {sorted(unexpected)}")
    missing = expected_keys - set(state_dict)
    if missing:
        raise ValueError(f"missing expected keys in checkpoint: {sorted(missing)}")

    converted: dict[str, torch.Tensor] = {}
    for i in range(n_hidden):
        converted[f"features.{3 * i}.weight"] = state_dict[f"model.{3 * i}.weight"]
        converted[f"features.{3 * i + 1}.weight"] = state_dict[f"model.{3 * i + 1}.weight"]
        converted[f"features.{3 * i + 1}.bias"] = state_dict[f"model.{3 * i + 1}.bias"]
    converted["classifier.weight"] = state_dict[f"model.{classifier_idx}.weight"]
    converted["classifier.bias"] = state_dict[f"model.{classifier_idx}.bias"]
    return converted


def _sha256sum(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_meshnet_gn_checkpoint(
    network: torch.nn.Module, path: str, expected_sha256: str | None = None
) -> torch.nn.Module:
    """Load a brainchop-models checkpoint file into `network` in place.

    `large_files.yml`'s `hash_val` is only checked once, at download time
    (see ci/utils.py's download_large_files); it says nothing about the file
    still on disk by the time a later `monai.bundle run` actually loads it.
    Passing `expected_sha256` (the same value large_files.yml pins) re-checks
    the file's digest here too, so a checkpoint that was replaced or
    corrupted after that initial download is rejected before its weights
    ever reach the network, rather than silently loaded.
    """
    if expected_sha256 is not None:
        actual_sha256 = _sha256sum(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"checkpoint at '{path}' has sha256 {actual_sha256}, expected {expected_sha256}")
    raw = load_safetensors(path, device="cpu")
    network.load_state_dict(convert_meshnet_gn_checkpoint(raw), strict=True)
    return network
