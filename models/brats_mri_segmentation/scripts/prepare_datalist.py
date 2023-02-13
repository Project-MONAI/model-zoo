import argparse
import glob
import json
import os

import monai
from sklearn.model_selection import train_test_split


def produce_sample_dict(line: str):
    names = os.listdir(line)
    seg, t1ce, t1, t2, flair = [], [], [], [], []
    for name in names:
        name = os.path.join(line, name)
        if "_seg.nii" in name:
            seg.append(name)
        elif "_t1ce.nii" in name:
            t1ce.append(name)
        elif "_t1.nii" in name:
            t1.append(name)
        elif "_t2.nii" in name:
            t2.append(name)
        elif "_flair.nii" in name:
            flair.append(name)

    return {"label": seg[0], "image": t1ce + t1 + t2 + flair}


def produce_datalist(dataset_dir: str):
    """
    This function is used to split the dataset.
    It will produce 200 samples for training, and the other samples are divided equally
    into val and test sets.
    """

    samples = sorted(glob.glob(os.path.join(dataset_dir, "*", "*"), recursive=True))
    datalist = []
    for line in samples:
        datalist.append(produce_sample_dict(line))
    train_list, other_list = train_test_split(datalist, train_size=200)
    val_list, test_list = train_test_split(other_list, train_size=0.5)

    return {"training": train_list, "validation": val_list, "testing": test_list}


def main(args):
    """
    split the dataset and output the data list into a json file.
    """
    data_file_base_dir = os.path.join(os.path.abspath(args.path), "training")
    output_json = args.output
    # produce deterministic data splits
    monai.utils.set_determinism(seed=123)
    datalist = produce_datalist(dataset_dir=data_file_base_dir)
    with open(output_json, "w") as f:
        json.dump(datalist, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path",
        type=str,
        default="/workspace/data/medical/brats2018challenge",
        help="root path of brats 2018 dataset.",
    )
    parser.add_argument(
        "--output", type=str, default="configs/datalist.json", help="relative path of output datalist json file."
    )
    args = parser.parse_args()

    main(args)
