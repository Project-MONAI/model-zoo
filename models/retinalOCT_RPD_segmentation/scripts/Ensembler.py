from pycocotools.coco import COCO
import pandas as pd
from torchvision.ops.boxes import box_iou,box_convert
import torch
import numpy as np
import os
import json
from tqdm import tqdm

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class Ensembler():
    def __init__(self,output_dir,dataset_name,grplist,iou_thresh,coco_gt_path=None,coco_instances_results_fname=None):
        self.output_dir = output_dir
        self.dataset_name=dataset_name
        self.grplist = grplist
        self.iou_thresh=iou_thresh
        self.n_detectors = len(grplist)
        
        if coco_gt_path is None:
            fname_gt = os.path.join(output_dir,dataset_name +"_coco_format.json")
        else:
            fname_gt = coco_gt_path

        if coco_instances_results_fname is None:
            fname_dt = "coco_instances_results.json"
        else:
            fname_dt = coco_instances_results_fname

        #load in ground truth (form image lists)    
        coco_gt = COCO(fname_gt)
        #populate detector truths
        dtlist = []
        for grp in grplist:
            fname = os.path.join(output_dir,grp,fname_dt)     
            dtlist.append(coco_gt.loadRes(fname))
            print('Successfully loaded {} into memory. {} instance detected.\n'.format(fname,len(dtlist[-1].anns)))
 
        self.coco_gt = coco_gt
        self.cats = [cat['id'] for cat in self.coco_gt.dataset['categories']]
        self.dtlist = dtlist
        self.results=[]

        print('Working with {} models, {} categories, and {} images.'.format(self.n_detectors,len(self.cats),len(self.coco_gt.imgs.keys())))

    def mean_score_nms(self):
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
        print('Computing mean score non-max suppression ensembling for {} images.'.format(len(self.coco_gt.imgs.keys())))
        for img in tqdm(self.coco_gt.imgs.keys()): 
            #print(img)
            df = pd.DataFrame() #a dataframe of detections
            obj_set = set() #a set of objects (frozensets)
            for i,coco_dt in enumerate(self.dtlist): #for each detector append predictions to df
                df = df.append(pd.DataFrame(coco_dt.imgToAnns[img]).assign(det=i),ignore_index=True)
            if not df.empty:
                for cat in self.cats: #for each category
                    dfcat = df[df['category_id']==cat]
                    ts = box_convert(torch.tensor(dfcat['bbox']),in_fmt='xywh',out_fmt='xyxy') #list of tensor boxes for cateogory                  
                    iou_bool = np.array((box_iou(ts,ts)>self.iou_thresh)) #compute IoU matrix and threshold
                    for i in range(len(dfcat)): #for each detection in that category
                        fset = frozenset(dfcat.index[iou_bool[i]])
                        obj_set.add(fset) #compute set of sets representing objects
                    #find overlapping sets

                    # for fs in obj_set: #for existing sets
                    #     if fs&fset: #check for
                    #         fsnew = fs.union(fset)
                    #         obj_set.remove(fs)
                    #         obj_set.add(fsnew)      
                    obj_set = nik_merge(obj_set)           
                    for s in obj_set:#for each detected objects, find winning box and assign score as mean of scores
                        dfset = dfcat.loc[list(s)]
                        mean_score = dfset['score'].sum()/max(self.n_detectors,len(s)) #allows for more detections than detectors
                        winning_box = dfset.iloc[dfset['score'].argmax()].to_dict()
                        winning_box['score']=mean_score
                        winning_list.append(winning_box)
        print('{} resulting instances from NMS'.format(len(winning_list)))
        self.results =  winning_list
        return self

    def save_coco_instances(self,fname = "coco_instances_results.json"):
        if self.results:
            with open(os.path.join(self.output_dir,fname),'w') as f:
                f.write(json.dumps(self.results,cls=NpEncoder))
                f.flush()

if __name__ == "__main__":
    ens = Ensembler('dev',["fold1", "fold2", "fold3", "fold4","fold5"],.2)
    ens.mean_score_nms()