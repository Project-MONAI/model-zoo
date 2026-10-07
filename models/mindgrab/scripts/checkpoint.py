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

"""Loads MindGrabNet weights directly from the upstream brainchop-models
checkpoint (https://github.com/neuroneural/brainchop-models,
meshnet/mindgrab/model.pth), instead of shipping a pre-renamed copy.

That checkpoint's key names come from tinygrad's MeshNet class in
brainchop-models' own tiny_meshnet.py: a flat `self.model` list, positionally
renamed from the original Catalyst training checkpoint by
tools/convert_catalyst_*.py. Every hidden block contributes only its Conv3d's
weight (GroupNorm there is parameter-free, and the activation is functional),
so the checkpoint has `model.<i>.0.weight` for each hidden block and
`model.<N>.weight` / `model.<N>.bias` for the final classifier conv.

This naming convention comes from that shared MeshNet class itself, not from
anything mindgrab-specific, so it is not something that can drift
independently without a repo-wide change upstream. `large_files.yml`'s
`hash_val` additionally guards against the file at that URL silently
changing.
"""

from __future__ import annotations

import hashlib
import re

import torch

__all__ = ["convert_meshnet_checkpoint", "load_meshnet_checkpoint"]

_HIDDEN_KEY_RE = re.compile(r"^model\.(\d+)\.0\.weight$")


def convert_meshnet_checkpoint(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a brainchop-models tinygrad-MeshNet-style state dict into
    MindGrabNet's layout. The hidden-block count is read from the checkpoint
    itself, not hardcoded, so this works for any checkpoint sharing the same
    positional convention.

    Raises ValueError if the checkpoint doesn't match the expected shape of
    that convention (missing/non-contiguous blocks, missing classifier, or
    extra keys), rather than silently loading something wrong.
    """
    hidden_indices = {int(m.group(1)) for key in state_dict if (m := _HIDDEN_KEY_RE.match(key))}
    if not hidden_indices:
        raise ValueError("no 'model.<i>.0.weight' keys found -- not a MeshNet-style checkpoint")

    n_hidden = max(hidden_indices) + 1
    if hidden_indices != set(range(n_hidden)):
        raise ValueError(f"expected contiguous hidden-block indices 0..{n_hidden - 1}, got {sorted(hidden_indices)}")

    classifier_weight_key = f"model.{n_hidden}.weight"
    classifier_bias_key = f"model.{n_hidden}.bias"
    if classifier_weight_key not in state_dict:
        raise ValueError(f"expected classifier weight at '{classifier_weight_key}'")

    expected_keys = {f"model.{i}.0.weight" for i in range(n_hidden)} | {classifier_weight_key, classifier_bias_key}
    unexpected = set(state_dict) - expected_keys
    if unexpected:
        raise ValueError(f"unexpected keys in checkpoint: {sorted(unexpected)}")

    converted = {f"features.{3 * i}.weight": state_dict[f"model.{i}.0.weight"] for i in range(n_hidden)}
    converted["classifier.weight"] = state_dict[classifier_weight_key]
    # classifier_bias_key, if present, is intentionally dropped: the reference
    # runtime never applies it (brainchopC's model_meta.json declares
    # classifier_bias: false), so it has no counterpart in MindGrabNet.
    return converted


def _sha256sum(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_meshnet_checkpoint(network: torch.nn.Module, path: str, expected_sha256: str | None = None) -> torch.nn.Module:
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
    raw = torch.load(path, map_location="cpu", weights_only=True)
    network.load_state_dict(convert_meshnet_checkpoint(raw), strict=True)
    return network
