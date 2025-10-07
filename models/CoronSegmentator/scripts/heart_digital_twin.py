# -*- coding: utf-8 -*-
"""
File  : heart_digital_twin.py
Author: John Y. Ke, MC. Chen, TY. Lin, YC. Chan
Copyright © 2025 Hon Hai Precision Industry Co.,Ltd. All rights reserved.
License: Apache License 2.0

Description:
This script performs <brief description of converting NII Files to USD Files>.
"""

import argparse
import gc
import json
import logging
import os
import shutil
import time
from pathlib import Path

from scripts.cardiac_segmentation.cardiac_seg import Auto3DSeg
from scripts.coronaryArtery_segmentation.coronaryArtery_seg import NNUnetPredictor
from scripts.usd_design.usd_create import USDCreator

from .download import download_and_verify

NII_GZ_EXT = ".nii.gz"
STL_EXT = ".stl"
USD_EXT = ".usd"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("HeartDigitalTwin")


def parseArg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", default="./configs/inference.json", help="The json file which define the input config."
    )

    return parser.parse_args()


def create_output_folder(output_dir, nii_file):
    """Create an output folder based on the NIfTI file name (removes the suffix)."""
    seg_folder = os.path.join(output_dir, nii_file[: -len(NII_GZ_EXT)])
    os.makedirs(seg_folder, exist_ok=True)
    return seg_folder


def run_cardiac_segmentation(nii_file_path, seg_folder):
    """Perform cardiac segmentation."""
    try:
        Auto3DSeg(nii_file_path, seg_folder).run()
        logger.info(f"Cardiac segmentation completed for {os.path.basename(nii_file_path)}")
    except Exception as e:
        logger.error(f"Cardiac segmentation failed: {str(e)}")
        raise


def run_coronary_artery_segmentation(nii_file_path, seg_folder):
    """Perform coronary artery segmentation."""
    try:
        NNUnetPredictor(nii_file_path, seg_folder).run()
        logger.info(f"Coronary artery segmentation completed for {os.path.basename(nii_file_path)}")
    except Exception as e:
        logger.error(f"Coronary artery segmentation failed: {str(e)}")
        raise


def convert_stl_to_usd(seg_folder, nii_file):
    """
    Convert STL files to USD format and create material properties.
    """
    stl_files = [f for f in os.listdir(seg_folder) if f.endswith(STL_EXT)]
    if not stl_files:
        raise FileNotFoundError("No STL files found for USD conversion")

    usd_path = os.path.join(seg_folder, nii_file.replace(NII_GZ_EXT, USD_EXT))
    if os.path.exists(usd_path):
        os.remove(usd_path)

    try:
        stl_full_paths = [os.path.join(seg_folder, f) for f in stl_files]

        USDCreator(stl_full_paths, usd_path).create_usd()
        logger.info(f"USD file created at {usd_path}")
    except Exception as e:
        logger.error(f"USD creation failed: {str(e)}")
        raise
    finally:
        # Force garbage collection after heavy processing
        gc.collect()


class CoroSegmentator_Pipeline:
    def __init__(self, input_params: dict):
        self.input_image = input_params["inputFile"]
        self.output_dir = input_params["outputDir"]
        self.nii_file = os.path.basename(self.input_image)

    def run(self):
        """
        Process a single NIfTI file: segment cardiac and coronary artery regions, and convert to USD format.
        """
        start_time = time.time()
        # Download the weights of models according to large_file.yml
        logger.info(f"Starting downloading weights of models.")
        download_and_verify()

        logger.info(f"Starting processing of {self.nii_file}")
        nii_path = self.input_image
        seg_folder = create_output_folder(self.output_dir, self.nii_file)
        try:
            # Perform cardiac and coronary artery segmentation
            run_cardiac_segmentation(nii_path, seg_folder)
            run_coronary_artery_segmentation(nii_path, seg_folder)
            # Explicit garbage collection after segmentation to free memory before USD conversion
            gc.collect()
            # Convert STL files to USD format
            convert_stl_to_usd(seg_folder, self.nii_file)
            # Clean intermediate files
            for f in os.listdir(seg_folder):
                file_path = os.path.join(seg_folder, f)
                if os.path.isfile(file_path) and not file_path.endswith(USD_EXT):
                    os.remove(file_path)
            # Copy Texture files to the output directory
            script_dir = Path(os.path.abspath(__file__)).parent
            usd_texture_folder = os.path.join(script_dir, "usd_design/textures")
            out_texture_folder = os.path.join(seg_folder, "textures")
            shutil.copytree(usd_texture_folder, out_texture_folder, dirs_exist_ok=True)

            # Log total processing time
            logger.info(f"Total processing time for {self.nii_file}: {time.time() - start_time:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Processing failed for {self.nii_file}: {str(e)}")
            # Remove the output folder if failed
            if os.path.exists(seg_folder):
                for f in os.listdir(seg_folder):
                    os.remove(os.path.join(seg_folder, f))
                os.rmdir(seg_folder)
            return False
        finally:
            gc.collect()
