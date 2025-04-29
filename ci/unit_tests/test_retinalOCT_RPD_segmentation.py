import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import yaml


class TestRPDInference(unittest.TestCase):
    def setUp(self):
        print(os.getcwd())
        # set the bundle root to the directory the test is being run from.
        self.bundle_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "retinalOCT_RPD_segmentation")
        )
        # Change the current working directory to bundle_root
        os.chdir(self.bundle_root)

        # Create a temporary directory for test data
        self.test_data_dir = tempfile.mkdtemp()
        self.extracted_dir = os.path.join(self.bundle_root, "sample_data")

        # create a dummy metadata.json file.
        metadata_file = os.path.join(self.test_data_dir, "metadata.json")
        metadata = {
            "version": "0.0.1",
            "schema": "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/meta_schema_20220324.json",
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # create output directory.
        self.output_dir = os.path.join(self.test_data_dir, "output")
        os.makedirs(self.output_dir)

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_data_dir)

    def test_inference_run(self):
        # Override configuration parameters
        override = {
            "args": {
                "extracted_dir": self.extracted_dir,
                "output_dir": self.output_dir,
                "run_extract": False,
                "create_dataset": True,
                "run_inference": True,
                "binary_mask": True,
                "binary_mask_overlay": True,
                "instance_mask_overlay": True,
                "dataset_name": "testDataset",
            }
        }

        # Load the original inference.yaml
        inference_yaml_path = "configs/inference.yaml"
        with open(inference_yaml_path, "r") as f:
            inference_yaml = yaml.safe_load(f)

        # Modify inference.yaml with override parameters.
        inference_yaml["args"].update(override["args"])

        # Create a new inference.yaml in the test_data_dir
        test_inference_yaml_path = os.path.join(self.test_data_dir, "inference.yaml")
        with open(test_inference_yaml_path, "w") as f:
            yaml.dump(inference_yaml, f)

        # Run the inference command using subprocess
        cmd = [
            sys.executable,  # Use the same Python interpreter
            "-m",
            "monai.bundle",
            "run",
            "inference",
            "--bundle_root",
            self.bundle_root,
            "--config_file",
            test_inference_yaml_path,  # Use the new file
            "--meta_file",
            os.path.join(self.test_data_dir, "metadata.json"),
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            self.fail(f"Inference command failed: {e}")

        # Add assertions to check the output
        # Check if output files were created
        output_files = os.listdir(self.output_dir)
        self.assertTrue(len(output_files) > 0)

        # Check for the COCO JSON file
        coco_file_found = glob.glob(os.path.join(self.output_dir, "**", "coco_instances_results.json"), recursive=True)
        print(coco_file_found)
        self.assertTrue(len(coco_file_found) == 6)

        # Check for the TIFF files.
        tiff_files_found = glob.glob(os.path.join(self.output_dir, "**", "*.tiff"), recursive=True)
        self.assertTrue(len(tiff_files_found) == 6)

        # Check for the html files.
        html_files_found = glob.glob(os.path.join(self.output_dir, "*.html"))
        self.assertTrue(len(html_files_found) == 2)

        # At least 10 RPD present in sample data
        dfvol = pd.read_html(os.path.join(self.output_dir, "dfvol_testDataset.html"))[0]
        self.assertTrue(dfvol["dt_instances"].sum().iloc[0] > 10)


if __name__ == "__main__":
    unittest.main()
