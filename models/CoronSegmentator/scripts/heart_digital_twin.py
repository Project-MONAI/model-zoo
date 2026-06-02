# -*- coding: utf-8 -*-
"""
File  : heart_digital_twin.py
Author: John Y. Ke, MC. Chen, TY. Lin, YC. Chan
Copyright © 2025 Hon Hai Precision Industry Co.,Ltd. All rights reserved.
License: Apache License 2.0

Description:
This script runs coronary artery segmentation on NIfTI CT files,
downloads required model weights, and prepares segmentation outputs
for downstream 3D digital twin workflows.
"""

import argparse
import gc
import logging
import os
import shutil
from pathlib import Path

import yaml
from monai.apps.utils import download_url
from scripts.coronaryArtery_segmentation.coronaryArtery_seg import NNUnetPredictor

BUNDLE_ROOT = Path(__file__).parent.parent
DOWNLOAD_CONFIG = os.path.join(BUNDLE_ROOT, "large_file.yml")
NII_GZ_EXT = ".nii.gz"
STL_EXT = ".stl"
USD_EXT = ".usd"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("HeartDigitalTwin")


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", default="./configs/inference.json", help="The json file which define the input config."
    )

    return parser.parse_args()


def create_output_folder(output_dir, nii_file):
    """
    Create an output folder based on the
    NIfTI file name (removes the suffix).
    """
    seg_folder = os.path.join(output_dir, nii_file[: -len(NII_GZ_EXT)])
    os.makedirs(seg_folder, exist_ok=True)
    return seg_folder


def run_coronary_artery_segmentation(nii_file_path, seg_folder):
    """Perform coronary artery segmentation."""
    try:
        NNUnetPredictor(nii_file_path, seg_folder).run()
        logger.info(f"{os.path.basename(nii_file_path)} segmentation completed.")
    except Exception as e:
        logger.error(f"Coronary artery segmentation failed: {str(e)}")
        raise


class CoroSegmentatorPipeline:
    def __init__(self, input_params: dict):
        self.input_image = input_params["inputFile"]
        self.output_dir = input_params["outputDir"]
        self.nii_file = os.path.basename(self.input_image)

    def run(self):
        """
        Process a single NIfTI file: segment cardiac and coronary artery
        regions, and convert to USD format.
        """
        # Download the weights of models according to large_file.yml
        with open(DOWNLOAD_CONFIG, "r") as file:
            download_config = yaml.safe_load(file)

        for file_info in download_config["large_files"]:
            rel_path = file_info.get("path")
            url = file_info.get("url")
            expected_hash = file_info.get("hash_val")

            destination_path = BUNDLE_ROOT / rel_path
            destination_dir = Path(destination_path).parent
            destination_dir.mkdir(parents=True, exist_ok=True)
            download_url(url, filepath=destination_path, hash_val=expected_hash)

        logger.info(f"Starting processing of {self.nii_file}")
        nii_path = self.input_image
        seg_folder = create_output_folder(self.output_dir, self.nii_file)
        try:
            run_coronary_artery_segmentation(nii_path, seg_folder)
            return True
        except Exception:
            logger.exception(f"Processing failed for {self.nii_file}")
            shutil.rmtree(seg_folder, ignore_errors=True)
            raise
        finally:
            gc.collect()
