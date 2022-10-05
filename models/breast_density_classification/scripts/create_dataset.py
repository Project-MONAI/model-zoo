import json
import os
import sys


def main():
    base_dir = "/home/gupta/Documents/MONAI-DEPLOY/model-zoo/models/breast_density_classification/sample_data"
    list_classes = ["A", "B", "C", "D"]

    output_list = []
    for _class in list_classes:
        data_dir = os.path.join(base_dir, _class)
        list_files = os.listdir(data_dir)
        if _class == "A":
            _label = [1, 0, 0, 0]
        elif _class == "B":
            _label = [0, 1, 0, 0]
        elif _class == "C":
            _label = [0, 0, 1, 0]
        elif _class == "D":
            _label = [0, 0, 0, 1]

        for _file in list_files:
            _out = {"image": os.path.join(data_dir, _file), "label": _label}
            output_list.append(_out)

    output_file = sys.argv[1]
    data_dict = {"Test": output_list}

    fid = open(output_file, "w")
    json.dump(data_dict, fid, indent=1)


if __name__ == "__main__":
    main()
