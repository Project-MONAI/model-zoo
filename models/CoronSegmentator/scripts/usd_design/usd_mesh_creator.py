# -*- coding: utf-8 -*-
"""
File  : usd_mesh_creator.py
Author: John Y. Ke, MC. Chen, TY. Lin, YC. Chan
Copyright © 2025 Hon Hai Precision Industry Co.,Ltd. All rights reserved.
License: Apache License 2.0

Description:
This script performs conversion of multiple STL files into a single USD file with optimized memory usage and performance.

"""

import concurrent.futures
import gc
import logging
import os

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, Vt
from scripts.usd_design.environment_create import EnvironmentSetup

logger = logging.getLogger("UsdMeshCreator")


class USDMeshCreator:
    """
    A class to convert 3D mesh files (e.g. STL) into USD format, apply materials and lighting, and save the results.
    """

    def __init__(self, input_files, output_file):
        """Initialize the converter with input STL files and an output USD file."""
        self.input_files = input_files
        self.output_file = output_file
        self.stage = None
        self.mesh_paths = []

    @staticmethod
    def _get_organ_name(file_path):
        """Extract organ name from file path."""
        return os.path.basename(file_path).replace(".stl", "")

    @staticmethod
    def load_mesh(input_file):
        """Loads a 3D mesh from the specified file."""
        try:
            return trimesh.load_mesh(input_file)
        except Exception as e:
            logger.error(f"Error loading mesh from {input_file}: {e}")
            return None

    def get_paths(self, input_file):
        """Returns the USD paths for the mesh depending on its file type."""
        organ_mesh = self._get_organ_name(input_file)
        if organ_mesh:
            return (f"/root/{organ_mesh}", f"/root/{organ_mesh}/_img_segmentation_{organ_mesh}")
        return None, None

    def create_usd_mesh(self, mesh, xform_path, mesh_path):
        """Creates a USD mesh from a trimesh object and adds it to the USD stage at the specified paths."""
        try:
            # Create a root transform for scaling and translating the object
            UsdGeom.Xform.Define(self.stage, xform_path)
            # Create the mesh
            usd_mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)

            # Batch set attributes using numpy conversions
            vertices = mesh.vertices.astype(np.float32)
            usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))

            faces = mesh.faces.flatten().astype(np.int32)
            usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))

            counts = np.full(len(mesh.faces), 3, dtype=np.int32)
            usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))

            self.mesh_paths.append(mesh_path)
            return True
        except Exception as e:
            logger.error(f"Error creating USD mesh at {mesh_path}: {e}")
            return False

    @staticmethod
    def calculate_scene_center(meshes):
        """Calculate the center of the scene based on mesh bounds."""
        if not meshes:
            return Gf.Vec3f(0, 0, 0)

        all_mins = np.array([mesh.bounds[0] for mesh in meshes])
        all_maxs = np.array([mesh.bounds[1] for mesh in meshes])
        min_vals = np.min(all_mins, axis=0)
        max_vals = np.max(all_maxs, axis=0)

        return Gf.Vec3f(*(min_vals + max_vals) / 2.0)

    def convert_stl_to_usd(self):
        """Converts input STL files to USD format and saves them to the specified output file."""
        try:
            self.stage = Usd.Stage.CreateNew(self.output_file)
            EnvironmentSetup(self.stage).create_environment()
            UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)

            root_prim = UsdGeom.Xform.Define(self.stage, "/root")
            root_xform = UsdGeom.Xformable(root_prim)

            anim_rotate_op = root_xform.AddRotateYOp(precision=UsdGeom.XformOp.PrecisionFloat, opSuffix="animation")
            translate_op = root_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat, "translate_root")
            root_xform.AddXformOp(UsdGeom.XformOp.TypeRotateX, UsdGeom.XformOp.PrecisionFloat, "rotate_root").Set(-90)

            # Load meshes in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                meshes = list(executor.map(self.load_mesh, self.input_files))
            # meshes = [self.load_mesh(file) for file in self.input_files]

            mesh_paths = []
            for input_file, mesh in zip(self.input_files, meshes):
                xform_path, mesh_path = self.get_paths(input_file)
                if self.create_usd_mesh(mesh, xform_path, mesh_path):
                    mesh_paths.append(mesh_path)

            if mesh_paths:
                bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
                total_range = None
                for mesh_path in mesh_paths:
                    mesh_prim = self.stage.GetPrimAtPath(mesh_path)
                    if not mesh_prim or not mesh_prim.IsValid():
                        continue
                    bbox = bbox_cache.ComputeWorldBound(mesh_prim)
                    current_range = bbox.ComputeAlignedRange()
                    if total_range is None:
                        total_range = current_range
                    else:
                        total_range.UnionWith(current_range)
                if total_range and not total_range.IsEmpty():
                    bbox_min = total_range.GetMin()
                    bbox_max = total_range.GetMax()

                    base_center = Gf.Vec3d(
                        (bbox_min[0] + bbox_max[0]) / 2, bbox_min[1], (bbox_min[2] + bbox_max[2]) / 2
                    )

                    translation = -base_center
                    translate_op.Set(Gf.Vec3f(translation))

            time_codes = [0, 96]
            anim_rotate_op.Set(0.0, time_codes[0])
            anim_rotate_op.Set(360.0, time_codes[1])


            self.stage.SetStartTimeCode(time_codes[0])
            self.stage.SetEndTimeCode(time_codes[1])

            # Save final USD file
            self.stage.GetRootLayer().Save()
            # Clean up references to help garbage collection
            meshes.clear()

            return self.mesh_paths
        except Exception as e:
            logger.error(f"Error converting STL to USD: {e}")
            self.stage = None
            return []
        finally:
            gc.collect()
