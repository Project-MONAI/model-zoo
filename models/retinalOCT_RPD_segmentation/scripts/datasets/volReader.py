#!/usr/bin/env python
#
# Aaron Y. Lee MD MSCI (University of Washington) Copyright 2019
#
# Code ported from Markus Mayer's excellent work (https://www5.cs.fau.de/research/software/octseg/)
# 
# Also thanks to who contributed to the original openVol.m in Markus's project
#   Radim Kolar, Brno University, Czech Republic
#   Kris Sheets, Retinal Cell Biology Lab, Neuroscience Center of Excellence, LSU Health Sciences Center, New Orleans


import struct, array, datetime, codecs
import numpy as np
from collections import OrderedDict

class volFile():
    def __init__(self, filename):
        """
        Parses Heyex Spectralis *.vol files.

        Args:
            filename (str): Path to vol file

        Returns: 
            volFile class

        """
        self.__parseVolFile(filename)

    @property
    def oct(self):
        """
        Retrieve OCT volume as a 3D numpy array.

        Returns:
            3D numpy array with OCT intensities as 'uint8' array

        """
        return self.wholefile["cScan"]

    @property
    def irslo(self):
        """
        Retrieve IR SLO image as 2D numpy array

        Returns:
            2D numpy array with IR reflectance SLO image as 'uint8' array.

        """
        return self.wholefile["sloImage"]

    @property
    def grid(self):
        """
        Retrieve the IR SLO pixel coordinates for the B scan OCT slices

        Returns:
            2D numpy array with the number of b scan images in the first dimension
            and x_0, y_0, x_1, y_1 defining the line of the B scan on the pixel 
            coordinates of the IR SLO image.

        """
        wf = self.wholefile
        grid = []
        for bi in range(len(wf["slice-headers"])):
            bscanHead = wf["slice-headers"][bi]
            x_0 = int(bscanHead["startX"] / wf["header"]["scaleXSlo"])
            x_1 = int(bscanHead["endX"] / wf["header"]["scaleXSlo"])
            y_0 = int(bscanHead["startY"] / wf["header"]["scaleYSlo"])
            y_1 = int(bscanHead["endY"] / wf["header"]["scaleYSlo"])
            grid.append([x_0, y_0, x_1, y_1])
        return grid

    def renderIRslo(self, filename, renderGrid=False):
        """
        Renders IR SLO image as a PNG file and optionally overlays grid of B scans

        Args:
            filename (str): filename to save IR SLO image
            renderGrid (bool): True will render red lines for the location of the B scans.

        Returns:
            None

        """
        from PIL import Image, ImageDraw
        wf = self.wholefile
        a = np.copy(wf["sloImage"])
        if renderGrid:
            a = np.stack((a,)*3, axis=-1)
            a = Image.fromarray(a)
            draw = ImageDraw.Draw(a)
            grid = self.grid
            for (x_0, y_0, x_1, y_1) in grid:
                draw.line((x_0,y_0, x_1, y_1), fill=(255,0,0), width=3)
            a.save(filename)
        else:
            Image.fromarray(a).save(filepre)

    def renderOCTscans(self, filepre = "oct", renderSeg=False):
        """
        Renders OCT images a PNG file and optionally overlays segmentation lines
        Also creates a CSV file of vol file features.

        Args:
            filepre (str): filename prefix. OCT Images will be named as "<prefix>_001.png"
            renderSeg (bool): True will render colored lines for the segmentation of the RPE, ILM, and NFL on the B scans.

        Returns:
            None

        """
        from PIL import Image
        wf = self.wholefile
        for i in range(wf["cScan"].shape[0]):
            a = np.copy(wf["cScan"][i])
            if renderSeg:
                a = np.stack((a,)*3, axis=-1)
                for li in range(wf["segmentations"].shape[0]):
                    for x in range(wf["segmentations"].shape[2]):
                        a[int(wf["segmentations"][li,i,x]),x, li] = 255

            Image.fromarray(a).save("%s_%03d.png" % (filepre, i))

    def __parseVolFile(self, fn, parseSeg=False):
        print(fn)
        wholefile = OrderedDict()
        decode_hex = codecs.getdecoder("hex_codec")
        with open(fn, "rb") as fin:
            header = OrderedDict()
            header["version"] = fin.read(12)
            header["octSizeX"] = struct.unpack("I", fin.read(4))[0] # lateral resolution
            header["numBscan"] = struct.unpack("I", fin.read(4))[0]
            header["octSizeZ"] = struct.unpack("I", fin.read(4))[0] # OCT depth
            header["scaleX"] = struct.unpack("d", fin.read(8))[0]
            header["distance"] = struct.unpack("d", fin.read(8))[0]
            header["scaleZ"] = struct.unpack("d", fin.read(8))[0]
            header["sizeXSlo"] = struct.unpack("I", fin.read(4))[0]
            header["sizeYSlo"] = struct.unpack("I", fin.read(4))[0]
            header["scaleXSlo"] = struct.unpack("d", fin.read(8))[0]
            header["scaleYSlo"] = struct.unpack("d", fin.read(8))[0]
            header["fieldSizeSlo"] = struct.unpack("I", fin.read(4))[0] # FOV in degrees
            header["scanFocus"] = struct.unpack("d", fin.read(8))[0]
            header["scanPos"] = fin.read(4)
            header["examTime"] = struct.unpack("=q", fin.read(8))[0] / 1e7
            header["examTime"] = datetime.datetime.utcfromtimestamp(header["examTime"] - (369*365.25+4) * 24*60*60) # needs to be checked
            header["scanPattern"] = struct.unpack("I", fin.read(4))[0]
            header["BscanHdrSize"] = struct.unpack("I", fin.read(4))[0]
            header["ID"] = fin.read(16)
            header["ReferenceID"] = fin.read(16)
            header["PID"] = struct.unpack("I", fin.read(4))[0] 
            header["PatientID"] = fin.read(21)
            header["unknown2"] = fin.read(3)
            header["DOB"] = struct.unpack("d", fin.read(8))[0] - 25569
            header["DOB"] = datetime.datetime.utcfromtimestamp(0) + datetime.timedelta(seconds=header["DOB"] * 24 * 60 * 60 ) # needs to be checked
            header["VID"] = struct.unpack("I", fin.read(4))[0]
            header["VisitID"] = fin.read(24)
            header["VisitDate"] = struct.unpack("d", fin.read(8))[0] - 25569
            header["VisitDate"] = datetime.datetime.utcfromtimestamp(0) + datetime.timedelta(seconds=header["VisitDate"] * 24 * 60 * 60) # needs to be checked
            header["GridType"] = struct.unpack("I", fin.read(4))[0]
            header["GridOffset"] = struct.unpack("I", fin.read(4))[0]

            wholefile["header"] = header
            fin.seek(2048)
            U = array.array("B")
            U.frombytes(fin.read(header["sizeXSlo"] * header["sizeYSlo"]))
            U = np.array(U).astype("uint8").reshape((header["sizeXSlo"],header["sizeYSlo"]))
            wholefile["sloImage"] = U

            sloOffset = 2048 + header["sizeXSlo"] * header["sizeYSlo"]
            octOffset = header["BscanHdrSize"] + header["octSizeX"] * header["octSizeZ"] * 4 
            bscans = []
            bscanheaders = []
            bscanqualities = []
            if parseSeg:
                segmentations = None
            for i in range(header["numBscan"]):
                fin.seek(16 + sloOffset + i * octOffset)
                bscanHead = OrderedDict()
                bscanHead["startX"] = struct.unpack("d", fin.read(8))[0]
                bscanHead["startY"] = struct.unpack("d", fin.read(8))[0]
                bscanHead["endX"] = struct.unpack("d", fin.read(8))[0]
                bscanHead["endY"] = struct.unpack("d", fin.read(8))[0]
                bscanHead["numSeg"] = struct.unpack("I", fin.read(4))[0]
                bscanHead["offSeg"] = struct.unpack("I", fin.read(4))[0]
                bscanHead["quality"] = struct.unpack("f", fin.read(4))[0]
                bscanHead["shift"] = struct.unpack("I", fin.read(4))[0]
                bscanheaders.append(bscanHead)
                bscanqualities.append(bscanHead["quality"])

                # extract OCT B scan data
                fin.seek(header["BscanHdrSize"] + sloOffset + i * octOffset)
                U = array.array("f")
                U.frombytes(fin.read(4 * header["octSizeX"] * header["octSizeZ"]))
                U = np.array(U).reshape((header["octSizeZ"],header["octSizeX"]))
                # remove out of boundary 
                v = struct.unpack("f", decode_hex('FFFF7F7F')[0])
                U[U == v] = 0
                # log normalize
                U = np.log(10000 * U + 1)
                U = (255. * (np.clip(U, 0, np.max(U)) / np.max(U))).astype("uint8")
                bscans.append(U)
                if parseSeg:
                    # extract OCT segmentations data
                    fin.seek(256 + sloOffset + i * octOffset)
                    U = array.array("f")
                    U.frombytes(fin.read(4 * header["octSizeX"] * bscanHead["numSeg"]))
                    U = np.array(U)
                    print(U.shape)
                    U[U == v] = 0.
                    if segmentations == None:
                        segmentations = []
                        for j in range(bscanHead["numSeg"]):
                            segmentations.append([])

                    for j in range(bscanHead["numSeg"]):
                        segmentations[j].append(U[j*header["octSizeX"]:(j+1) * header["octSizeX"]].tolist())
            wholefile["cScan"] = np.array(bscans)
            if parseSeg:
                wholefile["segmentations"] = np.array(segmentations)
            wholefile["slice-headers"] = bscanheaders
            wholefile["average-quality"] = np.mean(bscanqualities)
            self.wholefile = wholefile
        import csv
        from pathlib import Path, PurePath
        volFeatures = [PurePath(fn).name, wholefile["header"]["version"].decode("utf-8").rstrip('\x00'), wholefile["header"]["numBscan"], wholefile["header"]["octSizeX"], wholefile["header"]["octSizeZ"], wholefile["header"]["distance"], wholefile["header"]["scaleX"], \
                                wholefile["header"]["scaleZ"], wholefile["header"]["sizeXSlo"], wholefile["header"]["sizeYSlo"], wholefile["header"]["scaleXSlo"], wholefile["header"]["scaleYSlo"], wholefile["header"]["fieldSizeSlo"], \
                                wholefile["header"]["scanFocus"], wholefile["header"]["scanPos"].decode("utf-8").rstrip('\x00'), wholefile["header"]["examTime"], wholefile["header"]["scanPattern"], wholefile["header"]["BscanHdrSize"], \
                                wholefile["header"]["ID"].decode("utf-8").rstrip('\x00'), wholefile["header"]["ReferenceID"].decode("utf-8").rstrip('\x00'), wholefile["header"]["PID"], wholefile["header"]["PatientID"].decode("utf-8").rstrip('\x00'), wholefile["header"]["DOB"], wholefile["header"]["VID"], wholefile["header"]["VisitID"].decode("utf-8").rstrip('\x00'), wholefile["header"]["VisitDate"], \
                                wholefile["header"]["GridType"], wholefile["header"]["GridOffset"], wholefile["average-quality"]]
        output_dir = PurePath(fn).parent
        output_csv = output_dir.joinpath('vols.csv')
        if not Path(output_csv).exists():
            print("Creating vols.csv as it does not exist.")
            with open(output_csv, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["filename", "version", "numBscan", "octSizeX", "octSizeZ", "distance", "scaleX", \
                                "scaleZ", "sizeXSlo", "sizeYSlo", "scaleXSlo", "scaleYSlo", "fieldSizeSlo", \
                                "scanFocus", "scanPos", "examTime", "scanPattern", "BscanHdrSize", \
                                "ID", "ReferenceID", "PID", "PatientID", "DOB", "VID", "VisitID", "VisitDate", \
                                "GridType", "GridOffset", "Average Quality"])
        with open(output_csv, 'r', newline='') as file:
            existing_vols = csv.reader(file)
            for vol in existing_vols:
                if vol[0] == PurePath(fn).name:
                    print("Skipping,", PurePath(fn).name, "already present in vols.csv.")
                    return
        with open(output_csv, 'a', newline='') as file:
            print("Adding", PurePath(fn).name, "to vols.csv.")
            writer = csv.writer(file)
            writer.writerow(volFeatures)



    @property
    def fileHeader(self):
        """
        Retrieve vol header fields

        Returns:
            Dictionary with the following keys
                - version: version number of vol file definition
                - numBscan: number of B scan images in the volume
                - octSizeX: number of pixels in the width of the OCT B scan
                - octSizeZ: number of pixels in the height of the OCT B scan
                - distance: unknown
                - scaleX: resolution scaling factor of the width of the OCT B scan
                - scaleZ: resolution scaling factor of the height of the OCT B scan
                - sizeXSlo: number of pixels in the width of the IR SLO image
                - sizeYSlo: number of pixels in the height of the IR SLO image
                - scaleXSlo: resolution scaling factor of the width of the IR SLO image
                - scaleYSlo: resolution scaling factor of the height of the IR SLO image
                - fieldSizeSlo: field of view (FOV) of the retina in degrees
                - scanFocus: unknown
                - scanPos: Left or Right eye scanned
                - examTime: Datetime of the scan (needs to be checked)
                - scanPattern: unknown
                - BscanHdrSize: size of B scan header in bytes
                - ID: unknown
                - ReferenceID
                - PID: unknown
                - PatientID: Patient ID string
                - DOB: Date of birth
                - VID: unknown
                - VisitID: Visit ID string
                - VisitDate: Datetime of visit (needs to be checked)
                - GridType: unknown
                - GridOffset: unknown

        """
        return self.wholefile["header"]

    def bScanHeader(self, slicei):
        """
        Retrieve the B Scan header information per slice.

        Args:
            slicei (int): index of B scan

        Returns:
            Dictionary with the following keys
                - startX: x-coordinate for B scan on IR. (see getGrid)
                - startY: y-coordinate for B scan on IR. (see getGrid)
                - endX: x-coordinate for B scan on IR. (see getGrid)
                - endY: y-coordinate for B scan on IR. (see getGrid)
                - numSeg: 2 or 3 segmentation lines for the B scan
                - quality: OCT signal quality
                - shift: unknown

        """
        return self.wholefile["slice-headers"][slicei]

    def saveGrid(self, outfn):
        """
        Saves the grid coordinates mapping OCT Bscans to the IR SLO image to a text file. The text file
        will be a tab-delimited file with 5 columns: The bscan number, x_0, y_0, x_1, y_1 in pixel space
        scaled to the resolution of the IR SLO image.

        Args:
            outfn (str): location of where to output the file

        Returns:
            None

        """
        grid = self.grid
        with open(outfn, "w") as fout:
            fout.write("bscan\tx_0\ty_0\tx_1\ty_1\n")
            ri = 0
            for r in grid:
                r = [ri] + r
                fout.write("%s\n" % "\t".join(map(str, r)))
                ri += 1


