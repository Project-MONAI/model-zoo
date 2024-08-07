import argparse
import glob
import json
import logging
import os
import sys

import monai
from sklearn.model_selection import train_test_split


def produce_datalist_splits(datalist, splits: list = None, train_split: float = 0.80, valid_test_split: float = 0.50):
    """
    This function is used to split the dataset.
    It will produce "train_size" number of samples for training.
    """
    if splits is None:
        splits = ["test"]
    if "train" in splits:
        train_list, other_list = train_test_split(datalist, train_size=train_split)
        if "valid" in splits:
            val_list, test_list = train_test_split(other_list, train_size=valid_test_split)
            return {"training": train_list, "validation": val_list, "testing": test_list}
        else:
            return {"training": train_list, "testing": other_list}
    elif "valid" in splits:
        val_list, test_list = train_test_split(datalist, train_size=valid_test_split)
        return {"validation": val_list, "testing": test_list}
    else:
        return {"testing": datalist}


def keep_image_label_pairs_only(a_images, a_labels, i_folder, l_folder):
    image_names = [a.split("/")[-1] for a in a_images]
    label_names = [a.split("/")[-1] for a in a_labels]
    # Check if all_labels == all_images, if all_images < all_labels, truncate all_labels
    # image_set = set(image_names)
    # label_set = set(label_names)
    # labelmissing = image_set.difference(label_set)
    # Find names labels not in images
    # imagemissing = label_set.difference(image_set)
    # print('Data_path: ', a_images[0])
    # print('Data folder: ',a_images[0].split('/')[-2])
    # print('Labels missing for: ', len(labelmissing))
    # print('Images missing for: ', len(imagemissing))
    a_images = sorted([os.path.join(i_folder, a) for a in image_names if a in label_names])
    # Keep only labels that have a scan
    image_names = [a.split("/")[-1] for a in a_images]
    a_labels = sorted([os.path.join(l_folder, a) for a in label_names if a in image_names])
    return a_images, a_labels


def parse_files(images_folder, labels_folder, file_extension_pattern):
    logging.info(f"parsing files at: {os.path.join(images_folder, file_extension_pattern)}")
    all_images = sorted(glob.glob(os.path.join(images_folder, file_extension_pattern)))
    all_labels = sorted(glob.glob(os.path.join(labels_folder, file_extension_pattern)))
    return all_images, all_labels


def get_datalist(args, images_folder, labels_folder):
    file_extension_pattern = "*" + args.file_extension + "*"
    if type(images_folder) is list:
        all_images = []
        all_labels = []
        for ifolder, lfolder in zip(images_folder, labels_folder):
            a_images, a_labels = parse_files(ifolder, lfolder, file_extension_pattern)
            a_images, a_labels = keep_image_label_pairs_only(a_images, a_labels, ifolder, lfolder)
            all_images += a_images
            all_labels += a_labels
    else:
        all_images, all_labels = parse_files(images_folder, labels_folder, file_extension_pattern)
        all_images, all_labels = keep_image_label_pairs_only(all_images, all_labels, images_folder, labels_folder)

    logging.info("Length of all_images: {}".format(len(all_images)))
    logging.info("Length of all_labels: {}".format(len(all_labels)))

    datalist = [{"image": image_name, "label": label_name} for image_name, label_name in zip(all_images, all_labels)]

    # datalist = datalist[0 : args.limit] if args.limit else datalist
    logging.info(f"datalist length is {len(datalist)}")
    return datalist


def main(args):
    """
    split the dataset and output the data list into a json file.
    """
    data_file_base_dir = args.path
    output_json = args.output
    # produce deterministic data splits
    monai.utils.set_determinism(seed=123)
    datalist = get_datalist(args, data_file_base_dir, os.path.join(data_file_base_dir, args.labels_folder))
    datalist = produce_datalist_splits(datalist, args.splits, args.train_split, args.valid_test_split)
    with open(output_json, "w") as f:
        json.dump(datalist, f, ensure_ascii=True, indent=4)
    logging.info("datalist json file saved to: {}".format(output_json))


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
    parser.add_argument("--train_split", type=int, default=0.80, help="fraction of Training samples.")
    parser.add_argument("--valid_test_split", type=int, default=0.50, help="fraction of valid/test samples.")
    parser.add_argument("--splits", type=list, default=["test"], help="splits to use for train, valid, and test.")
    parser.add_argument("--file_extension", type=str, default="nii", help="file extension of images and labels.")
    parser.add_argument("--labels_folder", type=str, default="labels/final", help="labels sub folder name")

    args = parser.parse_args()

    main(args)
