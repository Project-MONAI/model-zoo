import json
import os

import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from torchvision.ops.boxes import box_convert, box_iou
from tqdm import tqdm


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types.

    This encoder handles NumPy-specific types that are not serializable by
    the default JSON library by converting them into standard Python types.
    """

    def default(self, obj):
        """Converts NumPy objects to their native Python equivalents.

        Args:
            obj (any): The object to encode.

        Returns:
            any: The JSON-serializable representation of the object.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class Ensembler:
    """A class to ensemble predictions from multiple object detection models.

    This class loads ground truth data and predictions from several models,
    performs non-maximum suppression (NMS) to merge overlapping detections,
    and saves the final ensembled results in COCO format.
    """

    def __init__(
        self, output_dir, dataset_name, grplist, iou_thresh, coco_gt_path=None, coco_instances_results_fname=None
    ):
        """Initializes the Ensembler.

        Args:
            output_dir (str): The base directory where model outputs and
                ensembled results are stored.
            dataset_name (str): The name of the dataset being evaluated.
            grplist (list[str]): A list of subdirectory names, where each
                subdirectory contains the prediction file from one model.
            iou_thresh (float): The IoU threshold for considering two bounding
                boxes as overlapping during NMS.
            coco_gt_path (str, optional): The full path to the ground truth
                COCO JSON file. If None, it's assumed to be in `output_dir`.
                Defaults to None.
            coco_instances_results_fname (str, optional): The filename for the
                COCO prediction files within each model's subdirectory.
                Defaults to "coco_instances_results.json".
        """
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.grplist = grplist
        self.iou_thresh = iou_thresh
        self.n_detectors = len(grplist)

        if coco_gt_path is None:
            fname_gt = os.path.join(output_dir, dataset_name + "_coco_format.json")
        else:
            fname_gt = coco_gt_path

        if coco_instances_results_fname is None:
            fname_dt = "coco_instances_results.json"
        else:
            fname_dt = coco_instances_results_fname

        # load in ground truth (form image lists)
        coco_gt = COCO(fname_gt)
        # populate detector truths
        dtlist = []
        for grp in grplist:
            fname = os.path.join(output_dir, grp, fname_dt)
            dtlist.append(coco_gt.loadRes(fname))
            print("Successfully loaded {} into memory. {} instance detected.\n".format(fname, len(dtlist[-1].anns)))

        self.coco_gt = coco_gt
        self.cats = [cat["id"] for cat in self.coco_gt.dataset["categories"]]
        self.dtlist = dtlist
        self.results = []

        print(
            "Working with {} models, {} categories, and {} images.".format(
                self.n_detectors, len(self.cats), len(self.coco_gt.imgs.keys())
            )
        )

    def mean_score_nms(self):
        """Performs non-maximum suppression by merging overlapping boxes.

        This method iterates through all images and categories, merging sets of
        overlapping bounding boxes from different detectors based on the IoU
        threshold. For each merged set, it calculates a mean score and selects
        the single box with the highest original score as the representative
        detection for the ensembled output.

        Returns:
            Ensembler: The instance itself, with the `self.results` attribute
                populated with the ensembled predictions.
        """

        def nik_merge(lsts):
            """Niklas B. https://github.com/rikpg/IntersectionMerge/blob/master/core.py"""
            sets = [set(lst) for lst in lsts if lst]
            merged = 1
            while merged:
                merged = 0
                results = []
                while sets:
                    common, rest = sets[0], sets[1:]
                    sets = []
                    for x in rest:
                        if x.isdisjoint(common):
                            sets.append(x)
                        else:
                            merged = 1
                            common |= x
                    results.append(common)
                sets = results
            return sets

        winning_list = []
        print(
            "Computing mean score non-max suppression ensembling for {} images.".format(len(self.coco_gt.imgs.keys()))
        )
        for img in tqdm(self.coco_gt.imgs.keys()):
            # print(img)
            dflist = []  # a dataframe of detections
            obj_set = set()  # a set of objects (frozensets)
            for i, coco_dt in enumerate(self.dtlist):  # for each detector append predictions to df
                dflist.append(pd.DataFrame(coco_dt.imgToAnns[img]).assign(det=i))
            df = pd.concat(dflist, ignore_index=True)
            if not df.empty:
                for cat in self.cats:  # for each category
                    dfcat = df[df["category_id"] == cat]
                    ts = box_convert(
                        torch.tensor(dfcat["bbox"]), in_fmt="xywh", out_fmt="xyxy"
                    )  # list of tensor boxes for cateogory
                    iou_bool = np.array((box_iou(ts, ts) > self.iou_thresh))  # compute IoU matrix and threshold
                    for i in range(len(dfcat)):  # for each detection in that category
                        fset = frozenset(dfcat.index[iou_bool[i]])
                        obj_set.add(fset)  # compute set of sets representing objects
                    # find overlapping sets

                    # for fs in obj_set: #for existing sets
                    #     if fs&fset: #check for
                    #         fsnew = fs.union(fset)
                    #         obj_set.remove(fs)
                    #         obj_set.add(fsnew)
                    obj_set = nik_merge(obj_set)
                    for s in obj_set:  # for each detected objects, find winning box and assign score as mean of scores
                        dfset = dfcat.loc[list(s)]
                        mean_score = dfset["score"].sum() / max(
                            self.n_detectors, len(s)
                        )  # allows for more detections than detectors
                        winning_box = dfset.iloc[dfset["score"].argmax()].to_dict()
                        winning_box["score"] = mean_score
                        winning_list.append(winning_box)
        print("{} resulting instances from NMS".format(len(winning_list)))
        self.results = winning_list
        return self

    def save_coco_instances(self, fname="coco_instances_results.json"):
        """Saves the ensembled prediction results to a JSON file.

        The output file follows the COCO instance format and can be used for
        further evaluation.

        Args:
            fname (str, optional): The filename for the output JSON file.
                Defaults to "coco_instances_results.json".
        """
        if self.results:
            with open(os.path.join(self.output_dir, fname), "w") as f:
                f.write(json.dumps(self.results, cls=NpEncoder))
                f.flush()


if __name__ == "__main__":
    # Example usage:
    # This assumes an 'output' directory with subdirectories 'fold1', 'fold2', etc.,
    # each containing a 'coco_instances_results.json' file.
    ens = Ensembler("dev", ["fold1", "fold2", "fold3", "fold4", "fold5"], 0.2)
    ens.mean_score_nms()
