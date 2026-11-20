# Copyright (c) 2026 The Authors
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE in the bundle root and https://polyformproject.org/licenses/noncommercial/1.0.0
"""Dataset for pre-computed pathology foundation-model embeddings.

The bundle's network does not see images: it sees one embedding vector per
foundation model (UNI, Virchow2, domain-adapted Phikon-v2) for each H&E tile /
ROI. Those embeddings are produced outside the bundle by
``scripts/extract_fm_embeddings.py`` in https://github.com/eliem08/paper_code
(the foundation models themselves are gated on Hugging Face and are NOT
distributed with this bundle -- see ``docs/data_license.txt``).

A datalist is a JSON list of entries::

    {
        "image_id": "BRACS_1228",
        "uni": "embeddings/BRACS_1228__uni.npy",
        "virchow2": "embeddings/BRACS_1228__virchow2.npy",
        "phikon_v2_ft": "embeddings/BRACS_1228__phikon_v2_ft.npy",
        "label": 2
    }

``label`` is the 3-class index (0 benign, 1 atypical, 2 malignant) and may be
omitted for unlabeled inference (it defaults to -1). Paths are resolved
relative to ``data_root``. Each embedding file is a float32 vector of the
expected dim (1024 / 2560 / 1024), stored as ``.npy``, ``.pt``, ``.json``
(a flat list of numbers) or ``.csv`` (one number per line or comma-separated).
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # imported as `scripts.embedding_dataset` by MONAI bundle configs
    from scripts.moe_network import CLASS_LABELS, EXPERT_DIMS, EXPERT_ORDER
except ImportError:  # imported / run directly from inside `scripts/`
    from moe_network import CLASS_LABELS, EXPERT_DIMS, EXPERT_ORDER

__all__ = ["EmbeddingListDataset", "load_datalist", "LABEL_TO_INDEX", "INDEX_TO_LABEL"]

LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASS_LABELS)}
INDEX_TO_LABEL = {i: name for name, i in LABEL_TO_INDEX.items()}


def load_datalist(datalist_file: str) -> list[dict[str, Any]]:
    """Load a datalist JSON file (a list, or a dict with a ``data`` key)."""
    with open(datalist_file) as f:
        entries = json.load(f)
    if isinstance(entries, dict):
        entries = entries.get("data", entries.get("test", []))
    return list(entries)


def _load_vector(path: str) -> torch.Tensor:
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".pt"):
        arr = torch.load(path, map_location="cpu", weights_only=True)
        arr = arr.numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    elif path.endswith(".json"):
        with open(path) as f:
            arr = np.asarray(json.load(f), dtype=np.float32)
    elif path.endswith(".csv"):
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    else:
        raise ValueError(f"unsupported embedding file (expected .npy, .pt, .json or .csv): {path}")
    return torch.as_tensor(np.asarray(arr, dtype=np.float32)).reshape(-1)


def prepare_embedding_batch(
    batchdata: dict[str, Any], device: str | torch.device | None = None, non_blocking: bool = False, **kwargs: Any
) -> tuple[Any, torch.Tensor | None]:
    """``prepare_batch`` for dict-of-embedding batches.

    MONAI's ``default_prepare_batch`` calls ``batch["image"].to(device)``,
    which fails when ``batch["image"]`` is a mapping of expert tensors
    (``{"uni": ..., "virchow2": ..., "phikon_v2_ft": ...}``). This moves every
    expert tensor (or a single concatenated tensor) to ``device`` instead.
    """
    image = batchdata["image"]
    if isinstance(image, dict):
        image = {k: v.to(device=device, non_blocking=non_blocking, **kwargs) for k, v in image.items()}
    else:
        image = image.to(device=device, non_blocking=non_blocking, **kwargs)
    label = batchdata.get("label")
    if isinstance(label, torch.Tensor):
        return image, label.to(device=device, non_blocking=non_blocking, **kwargs)
    return image, None


class EmbeddingListDataset(Dataset):
    """Torch dataset yielding ``{"image": {expert: tensor}, "label", "image_id"}``.

    ``data`` is either a datalist (list of entries, as returned by
    :func:`load_datalist`) or a path to a datalist JSON file.
    """

    def __init__(self, data: list[dict[str, Any]] | str, data_root: str = ".") -> None:
        self.data_root = data_root
        self.entries: list[dict[str, Any]] = load_datalist(data) if isinstance(data, str) else list(data)
        if not self.entries:
            raise ValueError("no entries found in datalist")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        image: dict[str, torch.Tensor] = {}
        for name in EXPERT_ORDER:
            rel_path = entry.get(name)
            if rel_path is None:
                raise KeyError(f"datalist entry {entry.get('image_id')} is missing embedding for {name!r}")
            path = rel_path if os.path.isabs(rel_path) else os.path.join(self.data_root, rel_path)
            vec = _load_vector(path)
            expected = EXPERT_DIMS[name]
            if vec.shape[0] != expected:
                raise ValueError(
                    f"{entry.get('image_id')}: {name} embedding has dim {vec.shape[0]}, expected {expected}"
                )
            image[name] = vec
        label = entry.get("label", -1)
        if isinstance(label, str):
            label = LABEL_TO_INDEX[label]
        return {
            "image": image,
            "label": torch.tensor(int(label), dtype=torch.int64),
            "image_id": str(entry.get("image_id", f"sample_{index}")),
        }
