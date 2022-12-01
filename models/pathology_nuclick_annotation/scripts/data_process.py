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
import argparse
import glob
import json
import logging
import os

from dataset import consep_nuclei_dataset

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=r"/workspace/data/CoNSeP",
        help="Input/Downloaded/Extracted dir for CoNSeP Dataset",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=r"/workspace/data/CoNSePNuclei",
        help="Output dir to store pre-processed data",
    )
    parser.add_argument("--crop_size", "-s", type=int, default=128, help="Crop size for each Nuclei")
    parser.add_argument("--limit", "-n", type=int, default=0, help="Non-zero value to limit processing max records")

    args = parser.parse_args()
    dataset_json = {}
    for f, v in {"Train": "training", "Test": "validation"}.items():
        logger.info("---------------------------------------------------------------------------------")
        if not os.path.exists(os.path.join(args.input, f)):
            logger.warning(f"Ignore {f} (NOT Exists in Input Folder)")
            continue

        logger.info(f"Processing Images/labels for: {f}")
        images_path = os.path.join(args.input, f, "Images", "*.png")
        labels_path = os.path.join(args.input, f, "Labels", "*.mat")
        images = sorted(glob.glob(images_path))
        labels = sorted(glob.glob(labels_path))
        ds = [{"image": i, "label": l} for i, l in zip(images, labels)]

        output_dir = os.path.join(args.output, f) if args.output else f
        crop_size = args.crop_size
        limit = args.limit

        ds_new = consep_nuclei_dataset(ds, output_dir, crop_size, limit=limit)
        logger.info(f"Total Generated/Extended Records: {len(ds)} => {len(ds_new)}")

        dataset_json[v] = ds_new

    ds_file = os.path.join(args.output, "dataset.json")
    with open(ds_file, "w") as fp:
        json.dump(dataset_json, fp, indent=2)
    logger.info(f"Dataset JSON Generated at: {ds_file}")


if __name__ == "__main__":
    main()
