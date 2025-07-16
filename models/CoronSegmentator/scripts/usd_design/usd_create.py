# -*- coding: utf-8 -*-
"""
File  : usd_create.py
Author: John Y. Ke, MC. Chen, TY. Lin, YC. Chan
Copyright © 2025 Hon Hai Precision Industry Co.,Ltd. All rights reserved.
License: Apache License 2.0

Description:
This script performs <brief introduction to converting multiple STL files into a single USD file>.

"""

import gc
import logging
import os

from scripts.usd_design.usd_mesh_creator import USDMeshCreator
from scripts.usd_design.usd_shader_applier import USDShaderApplier

logger = logging.getLogger("USDCreator")


class USDCreator:
    """
    A class to handle the creation of USD files from input mesh files and apply materials using a shader file.
    """

    def __init__(self, input_files, output_file):
        """nitialize the USDCreator class."""
        self.input_files = input_files
        self.output_file = output_file
        self.shader_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders.json")

    def create_usd(self):
        """Create a USD file from the input mesh files and apply materials."""
        logger.info(f"Starting <{self.input_files}> USD creation process.")
        try:
            # Convert the input STL mesh files to USD format.
            creator = USDMeshCreator(self.input_files, self.output_file)
            mesh_paths = creator.convert_stl_to_usd()

            # Apply materials from a shader file to the generated USD meshes.
            applier = USDShaderApplier(self.output_file, self.shader_file)
            applier.apply_materials_to_mesh(mesh_paths)
        except Exception as e:
            logger.error(f"USD creation failed: {str(e)}")
        finally:
            gc.collect()


def main(input_files, output_file):
    USDCreator(input_files, output_file).create_usd()


if __name__ == "__main__":
    in_files = [
        "./../Output/7 Spiral  0.75-0.4  Bv44  2 ALL_0000/heart.stl",
        "./../Output/7 Spiral  0.75-0.4  Bv44  2 ALL_0000/aorta.stl",
        "./../Output/7 Spiral  0.75-0.4  Bv44  2 ALL_0000/pulmonary_vein.stl",
        "./../Output/7 Spiral  0.75-0.4  Bv44  2 ALL_0000/atrial_appendage_left.stl",
        "./../Output/7 Spiral  0.75-0.4  Bv44  2 ALL_0000/superior_vena_cava.stl",
        "./../Output/7 Spiral  0.75-0.4  Bv44  2 ALL_0000/inferior_vena_cava.stl",
        "./../Output/7 Spiral  0.75-0.4  Bv44  2 ALL_0000/coronary_artery_0.stl",
        "./../Output/7 Spiral  0.75-0.4  Bv44  2 ALL_0000/coronary_artery_3.stl",
    ]
    out_file = "./../Output/7 Spiral  0.75-0.4  Bv44  2 ALL_0000/merge_7 Spiral  0.75-0.4  Bv44  2 ALL_0000.usd"

    main(in_files, out_file)
