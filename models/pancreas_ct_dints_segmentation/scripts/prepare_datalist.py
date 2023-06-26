import argparse
import glob
import json
import os

import monai
from sklearn.model_selection import train_test_split


def produce_sample_dict(line: str):
    return {"label": line, "image": line.replace("labelsTr", "imagesTr")}


def produce_datalist(dataset_dir: str, train_size: int = 196):
    """
    This function is used to split the dataset.
    It will produce "train_size" number of samples for training.
    """

    samples = sorted(glob.glob(os.path.join(dataset_dir, "labelsTr", "*"), recursive=True))
    samples = [_item.replace(os.path.join(dataset_dir, "labelsTr"), "labelsTr") for _item in samples]
    datalist = []
    for line in samples:
        datalist.append(produce_sample_dict(line))
    train_list, other_list = train_test_split(datalist, train_size=train_size)
    val_list, test_list = train_test_split(other_list, train_size=0.66)

    return {"training": train_list, "validation": val_list, "testing": test_list}


def main(args):
    """
    split the dataset and output the data list into a json file.
    """
    data_file_base_dir = args.path
    output_json = args.output
    # produce deterministic data splits
    monai.utils.set_determinism(seed=123)
    datalist = produce_datalist(dataset_dir=data_file_base_dir, train_size=args.train_size)
    with open(output_json, "w") as f:
        json.dump(datalist, f, ensure_ascii=True, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path",
        type=str,
        default="/workspace/data/msd/Task07_Pancreas",
        help="root path of MSD Task07_Pancreas dataset.",
    )
    parser.add_argument(
        "--output", type=str, default="dataset_0.json", help="relative path of output datalist json file."
    )
    parser.add_argument("--train_size", type=int, default=196, help="number of training samples.")
    args = parser.parse_args()

    main(args)
