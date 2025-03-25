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

from .volReader import volFile

script_dir = os.path.dirname(__file__)


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


# def extract_files(dirtoextract, extracted_path):
#     proceed = True
#     if ((os.path.isdir(extracted_path)) and (len(os.listdir(extracted_path))!=0)):
#         val = input(f'{extracted_path} exists and is not empty. Files may be overwritten. Proceed with extraction? (Y/N)')
#         proceed = bool(distutils.util.strtobool(val))
#     if proceed:
#         print(f"Extracting files from {dirtoextract} into {extracted_path}...")
#         files_to_extract = glob.glob(os.path.join(dirtoextract,'**/*.vol'),recursive=True)
#         for i,line in enumerate(tqdm(files_to_extract)):
#             fpath = line.strip('\n')
#             vol = volFile(fpath)
#             fpath = fpath.replace('\\','/')
#             path, scan_str = fpath.strip('.vol').rsplit('/',1)
#             extractpath = os.path.join(extracted_path,scan_str.replace('_','/'))
#             os.makedirs(extractpath,exist_ok=True)
#             preffix = os.path.join(extractpath, scan_str+'_oct')
#             vol.renderOCTscans(preffix)
#     else:
#         pass


def extract_files(dirtoextract, extracted_path, input_format):
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
            for i, line in enumerate(tqdm(files_to_extract)):
                fpath = line.strip("\n")
                vol = volFile(fpath)
                fpath = fpath.replace("\\", "/")
                path, scan_str = fpath.strip(".vol").rsplit("/", 1)
                extractpath = os.path.join(extracted_path, scan_str.replace("_", "/"))
                os.makedirs(extractpath, exist_ok=True)
                preffix = os.path.join(extractpath, scan_str + "_oct")
                vol.renderOCTscans(preffix)
        elif input_format == "dicom":
            keywords = ["SOPInstanceUID", "PatientID", "ImageLaterality", "SeriesDate"]
            list_of_dicts = []
            dirgen = glob.iglob(os.path.join(dirtoextract, "**/DICOMDIR"), recursive=True)

            for dsstr in dirgen:
                fs = FileSet(dcmread(dsstr))
                fsgenOPT = genOPTfs(fs)
                for fi in tqdm(fsgenOPT):
                    dd = dict()
                    # top level keywords
                    for key in keywords:
                        dd[key] = fi.get(key)

                    volpath = os.path.join(extracted_path, f"{fi.SOPInstanceUID}")
                    # volpath = os.path.join(extracted_path, f'{fi.PatientID}_{fi.ImageLaterality}_{fi.SeriesDate}') #path for volume
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
    dataset = []
    instances = 0
    wrong_poly = 0
    extracted_files = glob.glob(os.path.join(extracted_path, "**/*.[Pp][Nn][Gg]"), recursive=True)
    print("Generating dataset of images...")
    for fn in tqdm(extracted_files):
        fn_adjusted = fn.replace("\\", "/")
        imageid = fn_adjusted.split("/")[-1]
        im = cv2.imread(fn)
        dat = dict(file_name=fn_adjusted, height=im.shape[0], width=im.shape[1], image_id=imageid)
        dataset.append(dat)
    print(f"Found {len(dataset)} images")
    print(f"Found {instances} instances")
    print(f"Found {wrong_poly} too few vertices")
    return dataset


def genOPTfs(fs):
    for instance in fs.find(Modality="OPT"):
        ds = instance.load()
        yield ds
