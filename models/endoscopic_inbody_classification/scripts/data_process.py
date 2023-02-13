import argparse
import json
import os

train_rate = 0.6
val_rate = 0.2
test_rate = 0.2


def save_json(content, path, filename):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    dst_file_name = os.path.join(path, filename)
    with open(dst_file_name, "w+") as fp:
        json.dump(content, fp, indent=4, separators=(",", ":"))


def generate_labels(data_path, output_path):
    """
    Loading a model by name.

    Args:
        data_path: path to classification dataset, which must contain `inbody` and `outbody` directories.
        output_path: path to save labels
    """

    data_list = [os.path.join(root, x) for root, _, filenames in os.walk(data_path) for x in filenames if "jpg" in x]
    label_list = [int("outbody" in os.path.basename(os.path.dirname(x))) for x in data_list]
    data_label_json = [{"image": x, "label": y} for x, y in zip(data_list, label_list)]
    inbody_list = list(filter(lambda x: x["label"] == 0, data_label_json))
    outbody_list = list(filter(lambda x: not (x["label"] == 0), data_label_json))
    inbody_train_len = int(len(inbody_list) * train_rate)
    outbody_train_len = int(len(outbody_list) * train_rate)
    inbody_val_len = int(len(inbody_list) * (train_rate + val_rate))
    outbody_val_len = int(len(outbody_list) * (train_rate + val_rate))
    inbody_train_list = inbody_list[:inbody_train_len]
    outbody_train_list = outbody_list[:outbody_train_len]
    inbody_val_list = inbody_list[inbody_train_len:inbody_val_len]
    outbody_val_list = outbody_list[outbody_train_len:outbody_val_len]
    inbody_test_list = inbody_list[inbody_val_len:]
    outbody_test_list = outbody_list[outbody_val_len:]
    train_list = inbody_train_list + outbody_train_list
    val_list = inbody_val_list + outbody_val_list
    test_list = inbody_test_list + outbody_test_list
    save_json(train_list, out_path, "train_samples.json")
    save_json(val_list, out_path, "val_samples.json")
    save_json(test_list, out_path, "test_samples.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path to downloaded dataset.
    parser.add_argument(
        "--datapath",
        type=str,
        default=r"/workspace/data/endoscopic_inbody_classification",
        help="The root path of the inbody classification dataset.",
    )

    # path to save label json.
    parser.add_argument("--outpath", type=str, default=r"./label", help="The output path of labels.")

    args = parser.parse_args()
    data_path = args.datapath
    out_path = args.outpath

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    generate_labels(data_path, out_path)
