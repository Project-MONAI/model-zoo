import argparse
import json
import os
import sys


def main(base_dir: str, output_file: str):
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

    data_dict = {"Test": output_list}

    fid = open(output_file, "w")
    json.dump(data_dict, fid, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-base_dir", "--base_dir", default="sample_data", help="dir of dataset")
    parser.add_argument(
        "-output_file", "--output_file", default="configs/sample_image_data.json", help="output file name"
    )
    parser_args, _ = parser.parse_known_args(sys.argv)
    main(base_dir=parser_args.base_dir, output_file=parser_args.output_file)
