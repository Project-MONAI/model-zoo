# -*- coding: utf-8 -*-
"""
Author: John Y. Ke, MC. Chen, TY. Lin, YC. Chan
Copyright © 2025 Hon Hai Precision Industry Co.,Ltd. All rights reserved.
License: Apache License 2.0

Description:
This script performs automatic 3D cardiac segmentation using deep learning models and converts
the results to STL format for anatomical structures.
"""

import concurrent.futures
import gc
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813
from scripts.cardiac_segmentation.auto3dseg_segresnet_inference import auto3dseg_inference
from scripts.file_process.file_conversion import FileConversion

# Configure logging
logger = logging.getLogger("CardiacSeg")


class Auto3DSeg:
    """
    A class for performing automatic 3D cardiac segmentation and STL conversion.

    Attributes:
        input_path (str): Path to input medical image (NIfTI format)
        output_path (str): Directory for output STL files
    """

    def __init__(self, input_path, output_path):
        """Initialize the Auto3DSeg instance."""
        self.instance_id = uuid.uuid4().hex
        self.input_path = input_path
        self.output_path = output_path
        self._setup_paths()

    def _setup_paths(self):
        """Configure internal paths for model, scripts and temporary storage."""
        script_dir = Path(os.path.abspath(__file__)).parent
        self.model_path = os.path.join(script_dir.parent.parent, "models/cardiacModel.pt")  # Model file path
        self.segments = [
            {"label": 1, "name": "heart"},
            {"label": 2, "name": "aorta"},
            {"label": 3, "name": "pulmonary_vein"},
            {"label": 11, "name": "atrial_appendage_left"},
            {"label": 12, "name": "superior_vena_cava"},
            {"label": 13, "name": "inferior_vena_cava"},
        ]

    @staticmethod
    def create_directory(path):
        """Creates a directory if it does not exist."""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _convert_to_nrrd(input_path, output_dir):
        """Convert input image to NRRD format."""
        try:
            base_name = Path(input_path).stem.split(".nii")[0]
            nrrd_path = os.path.join(output_dir, f"{base_name}.nrrd")
            FileConversion.convert_nii_to_nrrd(input_path, nrrd_path)

            logger.info(f"Conversion completed: {nrrd_path}")
            return nrrd_path
        except Exception as e:
            # logger.error(f"File conversion failed: {str(e)}")
            raise RuntimeError(f"File conversion failed: {str(e)}") from e

    def _run_segmentation(self, input_nrrd, output_dir):
        """ "Execute the segmentation model inference."""
        seg_output_nrrd = os.path.join(output_dir, f"{Path(input_nrrd).stem}.nrrd")

        logger.info("Running cardiac segmentation")
        try:
            auto3dseg_inference(model_file=self.model_path, image_file=input_nrrd, result_file=seg_output_nrrd)
            return seg_output_nrrd
        except Exception as e:
            raise RuntimeError("auto3dseg process failed") from e

    def _process_single_segment(self, seg_image, segment, output_dir):
        """Process individual anatomical segment and convert to STL."""
        label = segment["label"]
        name = segment["name"]

        if name == "pulmonary_vein":
            return

        try:
            mask = seg_image == label
            # Create temporary NRRD
            temp_nrrd = os.path.join(output_dir, f"{name}.nrrd")
            sitk.WriteImage(mask, temp_nrrd)
            del mask

            # Convert to STL
            temp_stl = temp_nrrd.replace(".nrrd", ".stl")
            FileConversion.convert_nrrd_to_stl(temp_nrrd, temp_stl, gaussian_sigma=1.2)

            # Move to final output
            shutil.move(temp_stl, os.path.join(self.output_path, f"{name}.stl"))

            # Cleanup temporary files
            os.remove(temp_nrrd)
        except Exception as e:
            # logger.error(f"Error processing {name}: {str(e)}")
            raise e
        finally:
            gc.collect()

    def _process_segmentation_results(self, seg_nrrd_path, output_dir):
        """Process all segments in parallel and generate STL files."""
        try:
            seg_image = sitk.ReadImage(seg_nrrd_path)

            seg_array = np.unique(sitk.GetArrayFromImage(seg_image))
            unique_labels = set(seg_array)
            segments_to_process = [s for s in self.segments if s["label"] in unique_labels]
            logger.info(
                f"Processing {len(segments_to_process)} segments out of {len(self.segments)} \
                  defined segments for {os.path.basename(seg_nrrd_path)}"
            )
            workers = 4  # min(int(os.cpu_count() * 0.75), len(segments_to_process))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self._process_single_segment, seg_image, segment, output_dir)
                    for segment in self.segments
                ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result(timeout=300)
                    except TimeoutError as te:
                        logger.error(f"TimeoutError: {str(te)}")
                        future.cancel()
                    except Exception as e:
                        logger.error(f"processing task for {futures[future]} failed: {str(e)}")
            # for segment in self.segments:
            #    self._process_single_segment(seg_image, segment, output_dir)

            FileConversion.convert_nrrd_to_nii(
                seg_nrrd_path, os.path.join(self.output_path, f"{Path(seg_nrrd_path).stem}.nii")
            )

            os.remove(seg_nrrd_path)
            del seg_image, seg_array
            gc.collect()
        except Exception as e:
            raise RuntimeError(f"Result processing failed: {str(e)}") from e

    def run(self):
        """Execute the complete processing pipeline."""
        start_time = time.time()
        logger.info(f"Starting cardiac segmentation for {os.path.basename(self.input_path)}")

        self.create_directory(self.output_path)

        with (
            tempfile.TemporaryDirectory(prefix=f"CardiacSeg_Input_{self.instance_id}_") as input_tmp_dir,
            tempfile.TemporaryDirectory(prefix=f"CardiacSeg_Output_{self.instance_id}_") as output_tmp_dir,
        ):
            try:
                # Step 1: Convert input to NRRD
                input_nrrd = self._convert_to_nrrd(self.input_path, input_tmp_dir)
                # Step 2: Run segmentation
                seg_nrrd = self._run_segmentation(input_nrrd, output_tmp_dir)
                # Step 3: Process results
                self._process_segmentation_results(seg_nrrd, output_tmp_dir)
            except Exception as e:
                # logger.error(f"Processing failed: {str(e)}")
                raise e

        total_time = time.time() - start_time
        logger.info(
            f"Successfully completed all cardiac segmentation steps for {os.path.basename(self.input_path)} in {total_time:.2f}s"
        )
