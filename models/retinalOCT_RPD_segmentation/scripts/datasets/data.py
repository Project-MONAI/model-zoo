import distutils.util
import glob
import os
import shutil

import cv2
import pandas as pd
from PIL import Image
from pydicom import dcmread
from pydicom.fileset import FileSet
from tqdm import tqdm

from .volReader import VolFile

script_dir = os.path.dirname(__file__)


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


def extract_files(dirtoextract, extracted_path, input_format):
    """Extracts individual image frames from .vol or DICOM files.

    This function scans a directory for source files of a specified format
    and extracts them into a structured output directory as PNG images.
    It handles both .vol files and standard DICOM files. If the
    output directory already contains files, it will prompt the user
    before proceeding to overwrite them.

    Args:
        dirtoextract (str): The root directory to search for source files.
        extracted_path (str): The destination directory where the extracted
            PNG images will be saved.
        input_format (str): The format of the input files. Must be either
            "vol" or "dicom".
    """
    assert input_format in ["vol", "dicom"], 'Error: input_format must be "vol" or "dicom".'
    proceed = True
    if (os.path.isdir(extracted_path)) and (len(os.listdir(extracted_path)) != 0):
        val = input(
            f"{extracted_path} exists and is not empty. Files may be overwritten. Proceed with extraction? (Y/N)"
        )
        proceed = bool(distutils.util.strtobool(val))
    if proceed:
        print(f"Extracting files from {dirtoextract} into {extracted_path}...")
        if input_format == "vol":
            files_to_extract = glob.glob(os.path.join(dirtoextract, "**/*.vol"), recursive=True)
            for _, line in enumerate(tqdm(files_to_extract)):
                fpath = line.strip("\n")
                vol = VolFile(fpath)
                fpath = fpath.replace("\\", "/")
                path, scan_str = fpath.strip(".vol").rsplit("/", 1)
                extractpath = os.path.join(extracted_path, scan_str.replace("_", "/"))
                os.makedirs(extractpath, exist_ok=True)
                preffix = os.path.join(extractpath, scan_str + "_oct")
                vol.render_oct_scans(preffix)
        elif input_format == "dicom":
            keywords = ["SOPInstanceUID", "PatientID", "ImageLaterality", "SeriesDate"]
            list_of_dicts = []
            dirgen = glob.iglob(os.path.join(dirtoextract, "**/DICOMDIR"), recursive=True)

            for dsstr in dirgen:
                fs = FileSet(dcmread(dsstr))
                fsgenopt = gen_opt_fs(fs)
                for fi in tqdm(fsgenopt):
                    dd = dict()
                    # top level keywords
                    for key in keywords:
                        dd[key] = fi.get(key)

                    volpath = os.path.join(extracted_path, f"{fi.SOPInstanceUID}")
                    shutil.rmtree(volpath, ignore_errors=True)
                    os.mkdir(volpath)
                    n = fi.NumberOfFrames
                    for i in range(n):
                        fname = os.path.join(volpath, f"{fi.SOPInstanceUID}_oct_{i:03d}.png")
                        Image.fromarray(fi.pixel_array[i]).save(fname)
                        list_of_dicts.append(dd.copy())
            dfoct = pd.DataFrame(list_of_dicts, columns=keywords)
            dfoct.to_csv(os.path.join(extracted_path, "basic_meta.csv"))
    else:
        pass


def rpd_data(extracted_path):
    """Generates a dataset list from a directory of extracted image files.

    Scans a directory recursively for PNG images and creates a list of
    dictionaries, one for each image. This format is designed to be compatible
    with Detectron2's `DatasetCatalog` and can be adapted to hold ground truth instances for evaluation.

    Args:
        extracted_path (str): The root directory containing the extracted
            .png image files to be included in the dataset.

    Returns:
        list[dict]: A list where each dictionary represents an image and
            contains its file path, dimensions, and a unique ID.
    """
    dataset = []
    extracted_files = glob.glob(os.path.join(extracted_path, "**/*.[Pp][Nn][Gg]"), recursive=True)
    print("Generating dataset of images...")
    for fn in tqdm(extracted_files):
        fn_adjusted = fn.replace("\\", "/")
        imageid = fn_adjusted.split("/")[-1]
        im = cv2.imread(fn)
        dat = dict(file_name=fn_adjusted, height=im.shape[0], width=im.shape[1], image_id=imageid)
        dataset.append(dat)
    print(f"Found {len(dataset)} images")
    return dataset


def gen_opt_fs(fs):
    """A generator for finding and loading OPT modality DICOM datasets.

    This function filters a pydicom `FileSet` object for instances that have
    the modality set to "OPT" (Ophthalmic Tomography) and yields each one
    as a fully loaded pydicom dataset.

    Args:
        fs (pydicom.fileset.FileSet): The pydicom FileSet to search through.

    Yields:
        pydicom.dataset.FileDataset: A loaded DICOM dataset for each instance
            with the "OPT" modality found in the FileSet.
    """
    for instance in fs.find(Modality="OPT"):
        ds = instance.load()
        yield ds
