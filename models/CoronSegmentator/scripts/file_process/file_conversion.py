# -*- coding: utf-8 -*-
"""
File  : file_conversion.py
Author: John Y. Ke, MC. Chen, TY. Lin, YC. Chan
Copyright © 2025 Hon Hai Precision Industry Co.,Ltd. All rights reserved.
License: Apache License 2.0

Description:
This script performs conversion between different medical imaging formats including
NIfTI, NRRD, and STL for 3D visualization and processing.

"""
import gc
import logging
import os

import numpy as np
import SimpleITK as sitk
import trimesh
from skimage.measure import marching_cubes

# Configure logging
logger = logging.getLogger("FileConversion")


class FileConversion:
    """
    Class for handling various medical image format conversions, such as NIfTI to NRRD,
    and NIfTI/NRRD to STL format.
    """

    @staticmethod
    def convert_nrrd_to_nii(nrrd_file, nii_file):
        image = sitk.ReadImage(nrrd_file)
        # save as NIfTI
        sitk.WriteImage(image, nii_file)

    @staticmethod
    def convert_nii_to_nrrd(nii_gz_file, nrrd_file):
        """
        Converts a NIfTI (.nii.gz) file to NRRD format.

        Args:
            nii_gz_file (str): Path to the NIfTI file.
            nrrd_file (str): Path to save the converted NRRD file.
        """
        try:
            nii_image = sitk.ReadImage(nii_gz_file)
            sitk.WriteImage(nii_image, nrrd_file)
            logger.debug(f"Conversion successful: {os.path.basename(nrrd_file)}")
            return True
        except Exception as e:
            # logger.error(f"Error converting NIfTI to NRRD: {e}")
            raise RuntimeError(f"NIfTI to NRRD conversion failed: {str(e)}")

    @staticmethod
    def convert_nrrd_to_stl(nrrd_file_path, stl_file_path, gaussian_sigma=1.0):
        """
        Converts a NRRD file to STL format. It first reads the NRRD file, applies smoothing,
        and then uses the marching cubes algorithm to extract the mesh and save it as an STL.

        Args:
            nrrd_file_path (str): Path to the NRRD file.
            stl_file_path (str): Path to save the resulting STL file.
            gaussian_sigma (float): Sigma value for Gaussian smoothing (default: 1.0).
        """
        try:
            try:
                logger.debug(f"Reading NRRD file: {nrrd_file_path}")
                image = sitk.ReadImage(nrrd_file_path)
            except Exception as e:
                raise RuntimeError(f"Failed to read NRRD file: {str(e)}")

            # Apply Gaussian smoothing
            logger.debug(f"Applying Gaussian smoothing with sigma={gaussian_sigma}")
            try:
                smoothed_image = sitk.SmoothingRecursiveGaussian(image, sigma=gaussian_sigma)

                metadata = {
                    "spacing": np.array(smoothed_image.GetSpacing()),
                    "origin": np.array(smoothed_image.GetOrigin()),
                    "direction": np.array(smoothed_image.GetDirection()).reshape(3, 3),
                }
                del image
            except Exception as e:
                del image  # Free memory in case of error
                raise RuntimeError(f"Failed during image smoothing: {str(e)}")

            logger.debug("Converting image to numpy array")

            volume_data = sitk.GetArrayFromImage(smoothed_image)
            del smoothed_image
            logger.debug("Generating mesh using marching cubes algorithm")

            if np.all(volume_data == 0):

                dummy_vertices = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                dummy_faces = [[0, 1, 2]]
                mesh_obj = trimesh.Trimesh(vertices=dummy_vertices, faces=dummy_faces)

            else:
                verts, faces, _, _ = marching_cubes(volume_data, level=0.5)
                del volume_data

                logger.debug("Transforming vertices to physical coordinates")
                transform_matrix = np.diag(metadata["spacing"]) @ metadata["direction"].T
                physical_verts = verts[:, [2, 1, 0]] @ transform_matrix + metadata["origin"]
                del verts, transform_matrix

                logger.debug("Creating and processing mesh")
                mesh_obj = trimesh.Trimesh(vertices=physical_verts, faces=faces)
                del physical_verts, faces

            mesh_obj.process(validate=True)
            logger.debug(f"Exporting mesh to STL: {stl_file_path}")
            mesh_obj.export(stl_file_path)
            logger.info(f"Successfully converted {os.path.basename(nrrd_file_path)} to STL")

            return True
        except Exception as e:
            # logger.error(f"Error converting NRRD to STL: {e}")
            raise RuntimeError(f"NRRD to STL conversion failed: {str(e)}") from e
        finally:
            gc.collect()
