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
import copy
import glob
import json
import logging
import os
import pathlib

import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

logger = logging.getLogger(__name__)


def split_consep_nuclei_dataset(d, images_dir, labels_dir, centroids_dir, crop_size, min_area=80, min_distance=20):
    dataset_json = []
    logger.debug(f"Processing Image: {d['image']} => Label: {d['label']}")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(centroids_dir, exist_ok=True)

    # Image
    image_id = _get_basename_no_ext(d["image"])
    image = Image.open(d["image"]).convert("RGB")
    image_np = np.array(image)

    # Label
    m = loadmat(d["label"])
    instances = m["inst_map"]

    for nuclei_id, (class_id, (y, x)) in enumerate(zip(m["inst_type"], m["inst_centroid"]), start=1):
        x, y = (int(x), int(y))
        class_id = int(class_id)
        class_id = 3 if class_id in (3, 4) else 4 if class_id in (5, 6, 7) else class_id  # override

        item = _process_item(
            d=d,
            image_id=image_id,
            nuclei_id=nuclei_id,
            images_dir=images_dir,
            labels_dir=labels_dir,
            centroids_dir=centroids_dir,
            image_np=image_np,
            instances=instances,
            instance_idx=nuclei_id,
            crop_size=crop_size,
            image_size=image.size,
            class_id=class_id,
            centroid=(x, y),
            min_area=min_area,
            min_distance=min_distance,
        )
        if item:
            dataset_json.append(item)

    return dataset_json


def _process_item(
    d,
    image_id,
    nuclei_id,
    images_dir,
    labels_dir,
    centroids_dir,
    image_np,
    instances,
    instance_idx,
    crop_size,
    image_size,
    class_id,
    centroid,
    min_area,
    min_distance,
):
    bbox = _compute_bbox(crop_size, centroid, image_size)

    cropped_label_np = instances[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    cropped_label_np = np.array(cropped_label_np)

    signal = np.where(cropped_label_np == instance_idx, class_id, 0)
    if np.count_nonzero(signal) < min_area:
        return None

    x, y = centroid
    if x < min_distance or y < min_distance or (image_size[0] - x) < min_distance or (image_size[1] - y < min_distance):
        return None

    others = np.where(np.logical_and(cropped_label_np > 0, cropped_label_np != instance_idx), 255, 0)
    cropped_label_np = signal + others
    cropped_label = Image.fromarray(cropped_label_np.astype(np.uint8), None)

    cropped_image_np = image_np[bbox[0] : bbox[2], bbox[1] : bbox[3], :]
    cropped_image = Image.fromarray(cropped_image_np, "RGB")

    file_prefix = f"{image_id}_{class_id}_{str(instance_idx).zfill(4)}"
    image_file = os.path.join(images_dir, f"{file_prefix}.png")
    label_file = os.path.join(labels_dir, f"{file_prefix}.png")
    centroid_file = os.path.join(centroids_dir, f"{file_prefix}.txt")

    cropped_image.save(image_file)
    cropped_label.save(label_file)

    with open(centroid_file, "w") as fp:
        json.dump([centroid], fp)

    item = copy.deepcopy(d)
    item["nuclei_id"] = nuclei_id
    item["mask_value"] = class_id
    item["image"] = image_file
    item["label"] = label_file
    item["centroid"] = centroid
    return item


def _get_basename(path):
    head, tail = os.path.split(path)
    return tail or os.path.basename(head)


def _file_ext(name) -> str:
    suffixes = []
    for s in reversed(pathlib.Path(name).suffixes):
        if len(s) > 10:
            break
        suffixes.append(s)
    return "".join(reversed(suffixes)) if name else ""


def _get_basename_no_ext(path):
    p = _get_basename(path)
    e = _file_ext(p)
    return p.rstrip(e)


def _compute_bbox(patch_size, centroid, size):
    x, y = centroid
    m, n = size

    x_start = int(max(x - patch_size / 2, 0))
    y_start = int(max(y - patch_size / 2, 0))
    x_end = x_start + patch_size
    y_end = y_start + patch_size
    if x_end > m:
        x_end = m
        x_start = m - patch_size
    if y_end > n:
        y_end = n
        y_start = n - patch_size
    return x_start, y_start, x_end, y_end


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
        default=r"/workspace/data/CoNSePNuclick",
        help="Output dir to store pre-processed data",
    )
    parser.add_argument(
        "--crop_size",
        "-s",
        type=int,
        default=128,
        help="Crop size for each Nuclei",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=0,
        help="Non-zero value to limit processing max records",
    )

    args = parser.parse_args()
    for f in ["Train", "Test"]:
        logger.info("---------------------------------------------------------------------------------")
        if not os.path.exists(os.path.join(args.input, f)):
            logger.warning(f"Ignore {f} (NOT Exists in Input Folder)")
            continue

        logger.info(f"Processing Images/labels for: {f}")
        images_path = os.path.join(args.input, f, "Images", "*.png")
        labels_path = os.path.join(args.input, f, "Labels", "*.mat")
        images = list(sorted(glob.glob(images_path)))
        labels = list(sorted(glob.glob(labels_path)))
        ds = [{"image": i, "label": l} for i, l in zip(images, labels)]

        ds_new = []
        output_dir = args.output
        crop_size = args.crop_size
        images_dir = os.path.join(output_dir, f, "Images")
        labels_dir = os.path.join(output_dir, f, "Labels")
        centroids_dir = os.path.join(output_dir, f, "Centroids")
        limit = args.limit

        for d in tqdm(ds):
            ds_new.extend(split_consep_nuclei_dataset(d, images_dir, labels_dir, centroids_dir, crop_size))
            if 0 < limit < len(ds_new):
                break
        logger.info(f"Total Generated/Extended Records: {len(ds)} => {len(ds_new)}")


if __name__ == "__main__":
    main()
