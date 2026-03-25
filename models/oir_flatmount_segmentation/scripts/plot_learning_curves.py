#!/usr/bin/env python3
"""Aggregate and plot fold-wise learning curves for OIR training outputs."""

import argparse
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load_histories(kfold_dir: str) -> List[pd.DataFrame]:
    histories: List[pd.DataFrame] = []
    for fold in range(5):
        csv_path = os.path.join(kfold_dir, f"fold_{fold}", "training_history.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                df["fold"] = fold
                histories.append(df)
    return histories


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OIR k-fold learning curves.")
    parser.add_argument("--kfold_dir", type=str, required=True, help="Path containing fold_*/training_history.csv")
    parser.add_argument("--out", type=str, required=True, help="Output PNG path")
    args = parser.parse_args()

    histories = _load_histories(args.kfold_dir)
    if not histories:
        raise FileNotFoundError(f"No training_history.csv files found in: {args.kfold_dir}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for df in histories:
        fold = int(df["fold"].iloc[0])
        axes[0, 0].plot(df["epoch"], df["train_loss"], alpha=0.8, label=f"fold_{fold}")
        axes[0, 1].plot(df["epoch"], df["val_loss"], alpha=0.8, label=f"fold_{fold}")
        axes[1, 0].plot(df["epoch"], df["val_dice_nv"], alpha=0.8, label=f"fold_{fold}")
        axes[1, 1].plot(df["epoch"], df["val_dice_vo"], alpha=0.8, label=f"fold_{fold}")

    axes[0, 0].set_title("Train Loss")
    axes[0, 1].set_title("Validation Loss")
    axes[1, 0].set_title("Validation Dice - IVNV")
    axes[1, 1].set_title("Validation Dice - AVA")

    for ax in axes.flatten():
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)

    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Dice")
    axes[1, 1].set_ylabel("Dice")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved curves: {args.out}")


if __name__ == "__main__":
    main()

