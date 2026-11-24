# Copyright (c) 2026 The Authors
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE in the bundle root and https://polyformproject.org/licenses/noncommercial/1.0.0
"""Build a bundle datalist JSON from a directory of embedding files.

Embedding files must be named ``<image_id>__<expert>.npy`` (or ``.pt``,
``.json`` or ``.csv`` -- see ``embedding_dataset._load_vector``),
with ``<expert>`` one of ``uni``, ``virchow2``, ``phikon_v2_ft``. Labels are
read from an optional CSV with columns ``image_id,label`` (label as 0/1/2 or
benign/atypical/malignant); entries without a label row get no ``label`` key.

Example::

    python scripts/create_datalist.py \\
        --embeddings_dir sample_data/embeddings \\
        --output_file configs/sample_datalist.json \\
        --labels_csv sample_data/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

try:  # imported as `scripts.create_datalist` by MONAI bundle configs
    from scripts.embedding_dataset import LABEL_TO_INDEX
    from scripts.moe_network import EXPERT_ORDER
except ImportError:  # run directly from inside `scripts/`
    from embedding_dataset import LABEL_TO_INDEX
    from moe_network import EXPERT_ORDER


def build_datalist(embeddings_dir: str, labels_csv: str | None = None) -> list[dict]:
    labels: dict[str, int] = {}
    if labels_csv is not None:
        with open(labels_csv, newline="") as f:
            for row in csv.DictReader(f):
                raw = str(row["label"]).strip()
                labels[str(row["image_id"]).strip()] = LABEL_TO_INDEX[raw] if raw in LABEL_TO_INDEX else int(raw)

    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for fname in sorted(os.listdir(embeddings_dir)):
        if "__" not in fname:
            continue
        image_id, rest = fname.rsplit("__", 1)
        expert, _ext = os.path.splitext(rest)
        if expert in EXPERT_ORDER:
            grouped[image_id][expert] = os.path.abspath(os.path.join(embeddings_dir, fname))

    datalist: list[dict] = []
    for image_id in sorted(grouped):
        missing = [name for name in EXPERT_ORDER if name not in grouped[image_id]]
        if missing:
            raise ValueError(f"{image_id} is missing embeddings for: {missing}")
        entry: dict = {"image_id": image_id, **{name: grouped[image_id][name] for name in EXPERT_ORDER}}
        if image_id in labels:
            entry["label"] = labels[image_id]
        datalist.append(entry)
    return datalist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--labels_csv", default=None)
    args = parser.parse_args()
    datalist = build_datalist(args.embeddings_dir, args.labels_csv)
    with open(args.output_file, "w") as f:
        json.dump(datalist, f, indent=2)
    print(f"wrote {len(datalist)} entries to {args.output_file}")


if __name__ == "__main__":
    main()
