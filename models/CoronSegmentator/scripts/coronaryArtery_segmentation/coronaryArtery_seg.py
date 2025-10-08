#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
File  : coronaryArtery_seg.py
Author: John Y. Ke, MC. Chen, TY. Lin, YC. Chan
Copyright © 2025 Hon Hai Precision Industry Co.,Ltd. All rights reserved.
License: Apache License 2.0

Description:
This script performs segmentation of coronary arteries using nnUNetv2 and processes
the results for 3D visualization.
"""

import concurrent.futures
import gc
import logging
import os
import shutil
import subprocess
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy.spatial import cKDTree
from scripts.file_process.file_conversion import FileConversion
from stl import mesh

# Configure logging
logger = logging.getLogger("CoronaryArterySeg")


# Class representing information about each component (a group of connected triangles in the mesh)
class ComponentInfo:
    """
    Initializes a ComponentInfo instance with vertices.
    Computes the bounding box for the given vertices.
    """

    def __init__(self, vertices):
        self.vertices = vertices  # List of vertices for the component
        self.bbox = self.compute_bbox(vertices)  # Compute the bounding box for the component

    @staticmethod
    def compute_bbox(vertices):
        """
        Computes the bounding box for the given vertices.
        The bounding box is represented by two points (min_coords, max_coords) in 3D space.
        Optimized for performance with empty vertex handling.
        """
        if len(vertices) == 0:
            return (np.zeros(3), np.zeros(3))
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        return (min_coords, max_coords)


# Disjoint Set Union (DSU) class to keep track of connected components
class DSU:
    def __init__(self, components_info):
        """
        Initializes the DSU data structure.
        Sets up parent, rank (for union by rank), and stores the component info.
        """
        self.parent = list(range(len(components_info)))
        self.rank = [0] * len(components_info)
        self.info = components_info

    def find(self, x):
        """
        Finds the root of the set that x belongs to, using path compression.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """
        Unites the sets containing x and y. Performs union by rank.
        Also merges the vertices of the two components and updates their bounding box.
        Optimized to reduce memory usage during vertex merging.
        """
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.rank[x_root] < self.rank[y_root]:
            x_root, y_root = y_root, x_root
        self.parent[y_root] = x_root

        # Convert to list of tuples for hashability in set operations
        vertices_x = set(map(tuple, self.info[x_root].vertices))
        vertices_y = set(map(tuple, self.info[y_root].vertices))
        merged_vertices = vertices_x.union(vertices_y)
        merged_vertices = np.array([list(v) for v in merged_vertices])

        self.info[x_root] = ComponentInfo(merged_vertices)
        if self.rank[x_root] == self.rank[y_root]:
            self.rank[x_root] += 1


# Class to handle STL splitting based on distance threshold
class STLSplitter:
    """
    Initializes the STLSplitter instance.
    Reads the STL file and initializes relevant data structures.
    """

    def __init__(self, stl_file, distance_threshold=10, decimals=4):
        self.stl_file = stl_file
        self.distance_threshold = distance_threshold
        self.decimals = decimals
        self.original_mesh = mesh.Mesh.from_file(stl_file)
        self.num_triangles = len(self.original_mesh.vectors)
        self.vertex_map = defaultdict(list)
        self.initialize_vertex_map()

    def initialize_vertex_map(self):
        """
        Initializes the vertex map by iterating over all triangles in the mesh.
        Rounds the vertex coordinates to the specified number of decimals to avoid floating-point precision issues.
        """
        if self.num_triangles == 0:
            return

        vectors = self.original_mesh.vectors
        # Process in batches to reduce memory pressure
        batch_size = min(10000, self.num_triangles)  # Adjust batch size based on available memory
        for batch_start in range(0, self.num_triangles, batch_size):
            batch_end = min(batch_start + batch_size, self.num_triangles)
            for idx in range(batch_start, batch_end):
                triangle = vectors[idx]
                for vertex in triangle:
                    rounded = tuple(np.round(vertex, self.decimals))
                    self.vertex_map[rounded].append(idx)

    @staticmethod
    def bbox_distance(bbox_a, bbox_b):
        """
        Calculates the Euclidean distance between the bounding boxes of two components.
        """
        a_min, a_max = bbox_a
        b_min, b_max = bbox_b

        # Compute the distance between boxes
        deltas = np.maximum(0, np.maximum(a_min - b_max, b_min - a_max))

        return np.linalg.norm(deltas)

    @staticmethod
    def has_nearby_points(vertices_a, vertices_b, threshold):
        """
        Checks if two sets of vertices are close to each other within the specified threshold
        using a KD-Tree for efficient querying.
        """
        if len(vertices_a) == 0 or len(vertices_b) == 0:
            return False

        # Use the smaller set to build the KD-Tree for better performance
        if len(vertices_a) > len(vertices_b):
            vertices_a, vertices_b = vertices_b, vertices_a

        # Build KD-Tree only once
        tree = cKDTree(vertices_a)

        # Process vertices_b in batches to reduce memory usage
        batch_size = min(1000, max(100, len(vertices_b) // 10))  # Adaptive batch size

        # Use query_ball_tree for batch processing when vertices_b is large
        if len(vertices_b) > 10000:
            # Process in larger chunks for very large datasets
            for i in range(0, len(vertices_b), batch_size * 5):
                batch = vertices_b[i : i + batch_size * 5]
                # Use query_ball_point with r=threshold and return_length=True for early termination
                indices = tree.query_ball_point(batch, threshold, return_length=True)
                if any(indices):
                    return True
        else:
            # For smaller datasets, process point by point for early termination
            for i in range(0, len(vertices_b), batch_size):
                batch = vertices_b[i : i + batch_size]
                for point in batch:
                    # Early termination: return True as soon as we find any nearby point
                    if tree.query_ball_point(point, threshold, return_length=True):
                        return True
        return False

    def split(self):
        """
        Splits the STL mesh into independent components based on the distance threshold.
        Saves each component as a separate STL file.
        """
        # Step 1: Initial component identification using DSU
        logger.debug("Starting initial component identification")
        dsu_original = self.DSUOriginal(self.num_triangles)

        # Process vertex map in batches to reduce memory pressure
        batch_size = 10000  # Adjust based on available memory
        vertex_items = list(self.vertex_map.items())
        for i in range(0, len(vertex_items), batch_size):
            batch_end = min(i + batch_size, len(vertex_items))
            for _vert, tris in vertex_items[i:batch_end]:
                if len(tris) > 1:
                    root = tris[0]
                    for t in tris[1:]:
                        dsu_original.union(root, t)

        # Step 2: Group triangles by component
        logger.debug("Grouping triangles by component")
        components = defaultdict(list)
        for idx in range(self.num_triangles):
            root = dsu_original.find(idx)
            components[root].append(idx)
        initial_components = list(components.values())
        del components  # Free memory

        # Step 3: Extract vertices for each component
        logger.debug(f"Processing {len(initial_components)} initial components")
        component_info_list = []
        # Process components in batches to reduce memory pressure
        batch_size = 100  # Adjust based on available memory
        for i in range(0, len(initial_components), batch_size):
            batch_end = min(i + batch_size, len(initial_components))
            batch_components = initial_components[i:batch_end]

            for comp in batch_components:
                vertices = set()
                # Process triangles in smaller chunks
                chunk_size = 1000  # Adjust based on component size
                for j in range(0, len(comp), chunk_size):
                    chunk_end = min(j + chunk_size, len(comp))
                    for tri_idx in comp[j:chunk_end]:
                        triangle = self.original_mesh.vectors[tri_idx]
                        for vertex in triangle:
                            rounded_vertex = np.round(vertex, decimals=self.decimals)
                            vertices.add(tuple(rounded_vertex))

                # Convert to numpy array efficiently
                vertices = np.array([list(v) for v in vertices])
                component_info_list.append(ComponentInfo(vertices))

        # Step 4: Merge components based on distance threshold
        logger.debug("Merging components based on distance threshold")
        dsu = DSU(component_info_list)

        # Use a more efficient merging strategy
        roots = list({dsu.find(i) for i in range(len(component_info_list))})

        # Pre-compute bounding box distances to avoid redundant calculations
        merge_candidates = []
        for i in range(len(roots)):
            for j in range(i + 1, len(roots)):
                root_i = roots[i]
                root_j = roots[j]
                if dsu.find(root_i) != dsu.find(root_j):
                    info_i = dsu.info[root_i]
                    info_j = dsu.info[root_j]
                    dist_bbox = self.bbox_distance(info_i.bbox, info_j.bbox)
                    if dist_bbox <= self.distance_threshold:
                        merge_candidates.append((root_i, root_j, dist_bbox))

        # Sort candidates by distance for more efficient merging
        merge_candidates.sort(key=lambda x: x[2])

        # Perform merging
        for root_i, root_j, _ in merge_candidates:
            if dsu.find(root_i) != dsu.find(root_j):
                info_i = dsu.info[dsu.find(root_i)]
                info_j = dsu.info[dsu.find(root_j)]
                if self.has_nearby_points(info_i.vertices, info_j.vertices, self.distance_threshold):
                    dsu.union(root_i, root_j)

        # Step 5: Create final components
        logger.debug("Creating final component meshes")
        final_components = defaultdict(list)
        for i, comp in enumerate(initial_components):
            root = dsu.find(i)
            final_components[root].extend(comp)

        logger.info(f"Split into {len(final_components)} independent models.")

        # Step 6: Save each component as a separate STL file
        stl_files = []
        for comp_id, tri_indices in final_components.items():
            # Process in batches to reduce memory usage
            batch_size = min(10000, len(tri_indices))
            data = np.zeros(len(tri_indices), dtype=mesh.Mesh.dtype)

            for i in range(0, len(tri_indices), batch_size):
                end_idx = min(i + batch_size, len(tri_indices))
                batch_indices = tri_indices[i:end_idx]
                data["vectors"][i:end_idx] = self.original_mesh.vectors[batch_indices]
                data["normals"][i:end_idx] = self.original_mesh.normals[batch_indices]

            component_mesh = mesh.Mesh(data, remove_empty_areas=False)
            file_path = os.path.join(os.path.dirname(self.stl_file), f"coronary_artery_{comp_id}.stl")
            component_mesh.save(file_path)
            stl_files.append(file_path)

        return stl_files

    # Helper class for DSU (Disjoint Set Union) operations specific to the triangles
    class DSUOriginal:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size

        def find(self, x):
            """
            Finds the root of the set containing x using path compression.
            """
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            """
            Unites the sets containing x and y using union by rank.
            """
            x_root = self.find(x)
            y_root = self.find(y)
            if x_root == y_root:
                return
            if self.rank[x_root] < self.rank[y_root]:
                self.parent[x_root] = y_root
            else:
                self.parent[y_root] = x_root
                if self.rank[x_root] == self.rank[y_root]:
                    self.rank[x_root] += 1


class NNUnetPredictor:
    """
    A class to handle NN-UNet-based segmentation tasks, including preprocessing, prediction,
    and post-processing of segmentation results.
    """

    def __init__(self, input_path, output_path, dataset_id=66, configuration="3d_lowres"):
        """Initializes the NNUnetPredictor class with input paths, output paths, and model configuration."""
        # Fixed paths for the nnUNet environment
        self.instance_id = uuid.uuid4().hex
        self.dataset_id = dataset_id
        self.configuration = configuration

        self.base_name = Path(input_path).stem.split(".nii")[0]
        self.temp_dir = TemporaryDirectory(prefix=f"nnunet_{self.instance_id}_")
        try:
            script_path = Path(os.path.abspath(__file__))
            self.raw_path = os.path.join(self.temp_dir.name, "nnUNet/nnUNet_raw")
            self.preprocessed_path = os.path.join(script_path.parent, "nnUNet/nnUNet_preprocessed")
            self.results_path = os.path.join(script_path.parent.parent.parent, "models/nnUNet_results")
            self.model_weight = [
                os.path.join(script_path.parent.parent.parent, "models", f)
                for f in os.listdir(os.path.join(script_path.parent.parent.parent, "models"))
                if f.split(".")[-1] == "zip"
            ]  # check if any model weight exist

            # Path for coronary artery dataset and output
            self.img_path = os.path.join(self.raw_path, f"Dataset{dataset_id:03d}_CoronaryArtery")
            self.seg_path = os.path.join(self.temp_dir.name, "seg_output")

            # Input and output paths for CT data and results
            self.input_path = input_path
            self.output_path = output_path

            self.heart_nii_file = os.path.join(self.output_path, f"{self.base_name}.nii")
            self.coronary_nii_file = os.path.join(self.seg_path, f"{self.base_name}.nii.gz")
            self.coronary_npz_file = os.path.join(self.seg_path, f"{self.base_name}.npz")

            # Initialize directory structure
            self.create_directory(self.img_path)
            self.create_directory(self.seg_path)
            self.create_directory(self.output_path)

        except Exception as e:
            # logger.error(f"Failed to initialize paths: {str(e)}")
            raise RuntimeError(f"NNUnetPredictor initialization failed: {str(e)}") from e

    @staticmethod
    def create_directory(path):
        """Creates a directory if it does not exist."""
        os.makedirs(path, exist_ok=True)

    def _copy_input_file(self):
        """Copy input file to nnUNet's expected location."""
        try:
            shutil.copy(self.input_path, os.path.join(self.img_path, f"{self.base_name}_0000.nii.gz"))
        except Exception as e:
            # logger.error(f"Failed to copy input file: {str(e)}")
            raise RuntimeError(f"Failed to copy input file: {str(e)}") from e

    def _run_nnunet(self):
        """Runs the nnUNet prediction using the provided input and model configuration."""
        # Check if GPU is available and set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        env = os.environ.copy()
        env.update(
            {
                "nnUNet_raw": self.raw_path,
                "nnUNet_preprocessed": self.preprocessed_path,
                "nnUNet_results": self.results_path,
            }
        )

        # Construct the nnUNet prediction command
        command = [
            "nnUNetv2_predict",
            "-i",
            self.img_path,  # Input folder
            "-o",
            self.seg_path,  # Output folder
            "-d",
            f"{self.dataset_id:03d}",  # Dataset ID
            "-c",
            self.configuration,  # Configuration
            "-device",
            device.type,  # Device (CPU or GPU)
            "--disable_tta",
            "--save_probabilities",
        ]

        try:
            subprocess.run(
                command, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
            )
            logger.info(
                f"Coronary artery <{os.path.basename(self.input_path)}> segmentation inference has been completed."
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Coronary artery <{os.path.basename(self.input_path)}> segmentation inference failed with code {e.returncode}"
            )
            raise RuntimeError("nnUNet prediction failed") from e

    def set_model_weight(self, model_weight):
        """get model weight from zip file"""
        env = os.environ.copy()
        env.update(
            {
                "nnUNet_raw": self.raw_path,
                "nnUNet_preprocessed": self.preprocessed_path,
                "nnUNet_results": self.results_path,
            }
        )

        # Construct the nnUNet prediction command
        command = ["nnUNetv2_install_pretrained_model_from_zip", f"{model_weight}"]

        try:
            subprocess.run(
                command, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
            )
            logger.info(
                f"Coronary artery Segmentation: nnUnetv2 model weight was extracted successfully from {model_weight}."
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Coronary artery Segmentation: nnUnetv2 model weight from {model_weight}\
                failed to extract with code {e.returncode}"
            )
            raise RuntimeError("Load the model weight is failed") from e

    def _process_results(self):
        """Converts the NIfTI segmentation result to STL format and moves the result to the output directory."""

        def process_file(nii_path):
            """Process individual NIfTI file into split STLs."""
            try:
                stl_path = os.path.join(os.path.dirname(nii_path), f"coronary_artery_{self.instance_id}.stl")

                FileConversion.convert_nrrd_to_stl(nii_path, stl_path, gaussian_sigma=0.6)

                split_files = STLSplitter(stl_path, distance_threshold=10, decimals=4).split()
                for file in split_files:
                    # Move the STL file to the output directory
                    shutil.move(file, os.path.join(self.output_path, os.path.basename(file)))
            except Exception as e:
                logger.error(f"Error processing {nii_path}: {str(e)}. {traceback.print_exc()}")
                raise

        nii_files = [os.path.join(self.seg_path, f) for f in os.listdir(self.seg_path) if f.endswith(".nii.gz")]
        if not nii_files:
            logger.warning("No NIfTI files found in the segmentation output directory")
            return

        # Use thread pool for parallel processing with optimal number of workers
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(int(os.cpu_count() * 0.75), len(nii_files))
        ) as executor:
            futures = [executor.submit(process_file, nii_file) for nii_file in nii_files]
            # Wait for all tasks to complete and handle exceptions
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result(timeout=300)
                except TimeoutError as te:
                    logger.error(f"TimeoutError: {str(te)}")
                    future.cancel()
                except Exception as e:
                    logger.error(f"processing task for {futures[future]} failed: {str(e)}")

    def run(self):
        """Runs the entire pipeline: file handling, nnUNet inference, and result post-processing."""
        start_time = time.time()
        try:
            if self.model_weight:  # get the model weight from <model_weight.zip>
                for w in self.model_weight:
                    self.set_model_weight(w)

            logger.info(f"Starting coronary artery segmentation for {os.path.basename(self.input_path)}")

            # Step 1: Copy input file to nnUNet's expected location
            self._copy_input_file()
            # Step 2: Run nnUNet inference
            self._run_nnunet()
            gc.collect()
            # Step 3: Process segmentation results and convert them to STL
            self._process_results()

            elapsed_time = time.time() - start_time
            logger.info(
                f"Completed coronary artery segmentation pipeline \
                for {os.path.basename(self.input_path)} in {elapsed_time:.2f} seconds"
            )
        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            raise RuntimeError(f"Input file not found or accessible: {str(e)}") from e
        except PermissionError as e:
            logger.error(f"Permission error: {str(e)}")
            raise RuntimeError(f"Permission denied when accessing files: {str(e)}") from e
        except subprocess.CalledProcessError as e:
            logger.error(f"nnUNet process error (code {e.returncode}): {str(e)}")
            raise RuntimeError(f"nnUNet processing failed with code {e.returncode}") from e
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise RuntimeError(f"Coronary artery segmentation failed: {str(e)}") from e
        finally:
            # Clean up temporary directory
            try:
                self.temp_dir.cleanup()
                logger.debug(f"Cleaned up temporary directory: {self.temp_dir.name}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary directory: {str(cleanup_error)}")
