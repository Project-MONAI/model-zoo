#!/usr/bin/env python

"""
Utiltites for analyizing and visualizing model segmentations on dataset. 
Yelena Bagdasarova, Scott Song
"""

import os
import pickle
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import json
import sys

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import decode

import torch
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog)
import detectron2
import detectron2.utils.comm as comm
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator

import cv2
from PIL import Image
import matplotlib.pyplot as plt
plt.style.use('ybpres.mplstyle')
#plt.style.use('/data/ssong/detectron2-rpd-yb/detectron2-rpd-pkg/src/detectron2-rpd/ybpres.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


def grab_dataset(name):
    def f():
        return pickle.load( open( "datasets/"+name+".pk", "rb" ) )
    return f  

      
class OutputVis():

    def __init__(self,dataset_name,cfg=None,prob_thresh=0.5,pred_mode='model',pred_file=None,has_annotations=True, draw_mode = 'default'):
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.prob_thresh = prob_thresh
        self.data = DatasetCatalog.get(dataset_name)
        if pred_mode =='model':
            self.predictor = DefaultPredictor(cfg)
            self._mode = 'model'
        elif pred_mode =='file':
            with open(pred_file,'r') as f:
                self.pred_instances = json.load(f)
            self.instance_img_list = [p['image_id'] for p in self.pred_instances]
            self._mode = 'file'
        else:
            sys.exit('Invalid mode. Only "model" or "file" permitted.')
        self.has_annotations = has_annotations
        self.permitted_draw_modes = ['default','bw']
        self.set_draw_mode(draw_mode)
        self.font_size = 16 #28 for ARVO
        self.annotation_color = 'r'
        self.scale = 3.0

    def set_draw_mode(self,draw_mode):
        if draw_mode not in self.permitted_draw_modes:
            sys.exit('draw_mode must be one of the following: {}'.format(self.permitted_draw_modes))
        self.draw_mode = draw_mode

    def get_ori_image(self,ImgId):
        """[summary]

        Args:
            ImgId (str): Value of image_id in image data structure. 

        Returns:
            PIL.Image: Original image fed into the model scaled up by a factor of 3 for visualization.
        """
        dat = self.get_gt_image_data(ImgId) #gt
        im = cv2.imread(dat['file_name']) #input to model
        v_gt = Visualizer(im, MetadataCatalog.get(self.dataset_name), scale=self.scale)
        result_image = v_gt.output.get_image() #get original image
        img = Image.fromarray(result_image)
        return img

    def get_gt_image_data(self,ImgId):
        """Returns image ground truth image entry for corresponding to ImgId.

        Args:
            ImgId (str): Value of image_id in image data structure.

        Returns:
            dict: Dictionary for the image with corresponding ImgId.
        """
        gt_data = next(item for item in self.data if (item['image_id'] == ImgId))
        return gt_data    

    def produce_gt_image(self,dat,im):
        """Returns image im overlayed with ground truth instances found in dat if there are any. Depending on visualizer mode, the instances are separate colors or monochrome.

        Args:
            dat (dict): Dictionary for the image im containing ground truth annotations.
            im (numpy array): a numpy array of shape (H, W, C), where H and W correspond to
        the height and width of the image respectively. C is the number of color channels. The image is required to be in RGB format since that is a requirement of the Matplotlib library. The image is also expected to be in the range [0, 255].

        Returns:
            PIL.Image: The resulting original image overlayed with ground truth instances.
        """
        v_gt = Visualizer(im, MetadataCatalog.get(self.dataset_name), scale=self.scale) 
        if (self.has_annotations): #ground truth boxes and masks
            segs = [ddict['segmentation'] for ddict in dat['annotations']]
            if self.draw_mode is 'bw':
                BBoxes = None
                assigned_colors = [self.annotation_color]*len(segs)
            else: #default behavior
                bboxes = [ddict['bbox'] for ddict in dat['annotations']]
                BBoxes = detectron2.structures.Boxes(bboxes)
                BBoxes = detectron2.structures.BoxMode.convert(BBoxes.tensor,from_mode=1,to_mode=0) #0= XYXY, 1 = XYWH
                assigned_colors = None
            
            result_image = v_gt.overlay_instances(boxes=BBoxes,masks=segs,assigned_colors=assigned_colors, alpha=1.0).get_image()
        else:
            result_image = v_gt.output.get_image() #get original image if no annotations
        img = Image.fromarray(result_image)
        return img

    def produce_model_image(self,ImgId,dat,im):
        """Returns image im overlayed with instances predicted by the model. Depending on visualizer mode the model predicts on the image im or reads the predictions from file based on the given ImgId.

        Args:
            ImgId (str): Value of image_id in image data structure.
            dat (dict): Dictionary for the image im. Used for height and width parameters.
            im (numpy array): a numpy array of shape (H, W, C), where H and W correspond to
        the height and width of the image respectively. C is the number of color channels. The image is required to be in RGB format since that is a requirement of the Matplotlib library. The image is also expected to be in the range [0, 255].

        Returns:
            PIL.Image: The resulting original image overlayed with model-predicted instances.
        """
        v_dt = Visualizer(im, MetadataCatalog.get(self.dataset_name), scale=self.scale)
        v_dt._default_font_size = self.font_size

        #get predictions from model or file
        if self._mode=='model':
            outputs = self.predictor(im)["instances"].to("cpu")
        elif self._mode=='file':
            outputs = self.get_outputs_from_file(ImgId,(dat['height'],dat['width']))  
        outputs = outputs[outputs.scores>self.prob_thresh] #apply probability threshold to instances
        if self.draw_mode is 'bw':
            result_model = v_dt.overlay_instances(masks=outputs.pred_masks,assigned_colors=[self.annotation_color]*len(outputs), alpha=1.0).get_image()
        else: #default behavior
            result_model = v_dt.draw_instance_predictions(outputs).get_image()    
        img_model = Image.fromarray(result_model)
        return img_model

    def get_image(self,ImgId):
        """Returns the ground truth and model prediction overlays for the image corresponding to ImgId.

        Args:
            ImgId (str): Value of image_id in image data structure.

        Returns:
            (PIL.Image,PIL.Image): Tuple of PIL.images corresponding to ground truth and model prediction respectively.
        """
        dat = self.get_gt_image_data(ImgId) #gt
        im = cv2.imread(dat['file_name']) #input to model
        img = self.produce_gt_image(dat,im)
        img_model = self.produce_model_image(ImgId,dat,im)
        return img, img_model
  
    def get_outputs_from_file(self,ImgId,imgsize): 
        """For image with image_id ImgId, reads in and converts instances from coco format in self.pred_file to a detectron2 Instances structure required for the visulizer utility.

        Args:
            ImgId (str): Value of image_id in image data structure.
            imgsize (tuple): Height and width of the image in pixels.

        Returns:
            detectron2.structures.Instances: The correctly formated data structure to be used to the detectron2 visualizer utility.
        """

        pred_boxes = []
        scores = []
        pred_classes = []
        pred_masks = []
        for i,img in enumerate(self.instance_img_list):
            if img==ImgId:
                pred_boxes.append(self.pred_instances[i]['bbox'])
                scores.append(self.pred_instances[i]['score'])
                pred_classes.append(int(self.pred_instances[i]['category_id']))
                #pred_masks_rle.append(self.pred_instances[i]['segmentation'])
                pred_masks.append(decode(self.pred_instances[i]['segmentation']))
        BBoxes = detectron2.structures.Boxes(pred_boxes)
        pred_boxes = detectron2.structures.BoxMode.convert(BBoxes.tensor,from_mode=1,to_mode=0) #0= XYXY, 1 = XYWH
        inst_dict = dict(pred_boxes = pred_boxes,scores=torch.tensor(np.array(scores)),pred_classes=torch.tensor(np.array(pred_classes)),pred_masks = torch.tensor(np.array(pred_masks)).to(torch.bool))#pred_masks_rle=pred_masks_rle)
        outputs = detectron2.structures.Instances(imgsize,**inst_dict)
        return outputs

    @staticmethod
    def height_crop_range(im,height_target=256):
        """Find the range of pixels in the height dimension spanning the height height_target which contain the brightest regions of the image. 

        Args:
            im (numpy array): a numpy array of shape (H, W, C)
            height_target (int, optional): The desired span of the image height. Defaults to 256.

        Returns:
            range: The starting and stopping pixels for cropping the image height.
        """
        yhist = im.sum(axis=1) #integrate over width of image
        mu = np.average(np.arange(yhist.shape[0]),weights = yhist)
        h1 = int(np.floor(mu-height_target/2)) #inclusive
        h2 = int(np.ceil(mu+height_target/2)) #exclusive
        if h1<0:
            h1 = 0
            h2 = height_target
        if h2>yhist.shape[0]:
            h2 = yhist.shape[0]
            h1 = h2-height_target
        return range(h1,h2)

    def output_to_pdf(self,ImgIds,outname,dfimg=None):
        """Create pdf with name outname displaying ground truth and model prediction overlays for image ids listed in ImgIds.

        Args:
            ImgIds (list(str)): List of image_id values to output. 
            outname (str): path name for pdf    
            dfimg (pandas.DataFrame, optional): A dataframe of stats to display for each image. Defaults to None.
        """

        gtstr = ''
        dtstr = ''
        
        if dfimg is not None:
            gtcols = dfimg.columns[['gt_' in col for col in dfimg.columns]]
            dtcols = dfimg.columns[['dt_' in col for col in dfimg.columns]]
            
        with PdfPages(outname) as pdf:
            for imgid in tqdm(ImgIds):
                img, img_model = self.get_image(imgid)
                #pdb.set_trace()
                crop_range = self.height_crop_range(np.array(img.convert('L')),height_target=256*self.scale)
                img = np.array(img)[crop_range]
                img_model = np.array(img_model)[crop_range]
 
                fig, ax = plt.subplots(2,1,figsize=[22,10],dpi=200)
                ax[0].imshow(img)
                ax[0].set_title(imgid+' Ground Truth')
                ax[0].set_axis_off()
                ax[1].imshow(img_model)
                ax[1].set_title(imgid+' Model Prediction')
                ax[1].set_axis_off()
                if dfimg is not None: #annotate with provided stats
                    gtstr = ['{:s}={:.2f}'.format(col,dfimg.loc[imgid,col]) for col in gtcols]
                    ax[0].text(0,.05*(ax[0].get_ylim()[0]),gtstr,color='white',fontsize=14)
                    dtstr = ['{:s}={:.2f}'.format(col,dfimg.loc[imgid,col]) for col in dtcols]                
                    ax[1].text(0,.05*(ax[1].get_ylim()[0]),dtstr,color='white',fontsize=14)                   
                pdf.savefig(fig)
                plt.close(fig)

    def save_imgarr_to_tiff(self,imgs,outname):
        """Save array of images in stacked tiff format (one image per page).

        Args:
            imgs (array(PIL.Images)): An array of PIL.Images to save.
            outname (str): Path name to save to.
        """
        if len(imgs) > 1:
            imgs[0].save(outname,dpi=(400,400), tags = "", compression = None, save_all=True, append_images=imgs[1:])
        else:
            imgs[0].save(outname) 

    def output_ori_to_tiff(self,ImgIds,outname):
        """Save list of original images corresponding to ImgIds in stacked tiff format.

        Args:
            ImgIds (list(str)): A list of image_ids for images to save.
            outname (str): Path name to save to.
        """
        imgs = [] 
        for imgid in tqdm(ImgIds):
            img_ori = self.get_ori_image(imgid) #PIL Image
            imgs.append(img_ori)
        self.save_imgarr_to_tiff(imgs,outname)

    def output_pred_to_tiff(self,ImgIds,outname,pred_only=False):
        """Save list of images overlayed with the model predictions in stacked tiff format.     

        Args:
            ImgIds (list(str)): A list of image_ids for images to save.
            outname (str): Path name to save to.
        """
        imgs = self.output_pred_to_list(ImgIds,pred_only)
        self.save_imgarr_to_tiff(imgs,outname)

    def output_pred_to_list(self,ImgIds,pred_only=False):
        """Return list of images overlayed with the model predictions.     

        Args:
            ImgIds (list(str)): A list of image_ids for images to save.
            outname (str): Path name to save to.
        """
        imgs = [] 
        for imgid in tqdm(ImgIds):
            dat = self.get_gt_image_data(imgid) #gt
            if pred_only:
                im = np.zeros((dat['height'],dat['width'],3)) #blank image for overlay
                assert self._mode == 'file', 'pred_mode must be "file" when pred_only flage is set to True.' #fix this later
            else:    
                im = cv2.imread(dat['file_name']) #input to model
            img_dt = self.produce_model_image(imgid,dat,im)
            imgs.append(img_dt)
        return imgs

    def output_all_to_tiff(self,ImgIds,outname):
        """Save list of images (original, ground truth overlay, and model prediction overlay) to stacked tiff format. 

        Args:
            ImgIds (list(str)): A list of image_ids for images to save.
            outname (str): Path name to save to.
        """
        imgs = []
        for imgid in tqdm(ImgIds):
            img_gt, img_dt = self.get_image(imgid)
            img_ori = self.get_ori_image(imgid)
            hcrange = list(self.height_crop_range(np.array(img_ori.convert('L')),height_target=256*self.scale))
            img_result = Image.fromarray(np.concatenate((np.array(img_ori.convert('RGB'))[hcrange,:],np.array(img_gt)[hcrange,:],np.array(img_dt)[hcrange])))
            imgs.append(img_result)
        self.save_imgarr_to_tiff(imgs,outname)

    def get_enface_dt(self,grp,scan_height,scan_width,scan_spacing):
        """Return enface perspective of model predictions for a single scan volume whose imgids are listed in the index of grp.

        Args:
            grp (pandas.DataFrame): Dataframe containing images from a single scan volume indexed by ImageId.
            scan_height (int): Image height in pixels.
            scan_width (int): Image width in pixels.
            scan_spacing (float): The spacing between scan centers in pixels.

        Returns:
            np.array: Numpy array of dimension [scan_spacing*number of scans,scan_width,3]. 
        """
        grp = grp.sort_index()
        nscans = len(grp)
        enface_height = int(np.ceil((nscans-1)*scan_spacing))
        enface = np.zeros((enface_height,scan_width,3),dtype=int)
        for i,imgid in enumerate(grp.index):
            pos = int(np.clip(np.floor(scan_spacing*i),0,scan_width-1)) #vertical enface position


            outputs = self.get_outputs_from_file(imgid,(scan_height,scan_width))
            outputs = outputs[outputs.scores>self.prob_thresh]
            instances = outputs.pred_boxes[:,(0,2)].round().clip(0,scan_width-1).to(np.int)

            for inst in instances:
                try:
                    enface[max(pos-4,0):min(pos+4,scan_width-1),inst[0]:inst[1]]=np.array([255,255,255])#random_color(rgb = True)
                except(IndexError):
                    print( pos, inst[0],inst[1])
        return enface 

    def get_enface_gt(self,grp,scan_height,scan_width,scan_spacing):
        """Return enface perspective of ground truth annotations for a single scan volume whose imgids are listed in the index of grp.

        Args:
            grp (pandas.DataFrame): Dataframe containing images from a single scan volume indexed by ImageId.
            scan_height (int): Image height in pixels.
            scan_width (int): Image width in pixels.
            scan_spacing (float): The spacing between scan centers in pixels.

        Returns:
            np.array: Numpy array of dimension [scan_spacing*number of scans,scan_width,3]. 
        """
        grp = grp.sort_index()
        nscans = len(grp)
        enface_height = int(np.ceil((nscans-1)*scan_spacing))
        enface = np.zeros((enface_height,scan_width,3),dtype=int)
        if not self.has_annotations:
            enface[:,:] = np.array([100,100,100])

        else:
            #minx = scan_width
            for i,imgid in enumerate(grp.index):
                pos = int(np.clip(np.floor(scan_spacing*i),0,scan_width-1))
                instances =  self.get_gt_image_data(imgid)['annotations']
                for inst in instances:
                    x1= inst['bbox'][0]
                    #minx = min(minx,x1)
                    x2 = x1 + inst['bbox'][2]
                    try:
                        enface[max(pos-4,0):min(pos+4,scan_width-1),x1:x2]=np.array([255,255,255])#random_color(rgb = True)
                    except(IndexError):
                        print( pos, x1,x2)
        return enface

    def compare_enface(self,grp,name,scan_height,scan_width,scan_spacing):
        """Return figure comparing detector enface perspective of model prediction with ground truth.

        Args:
            grp (pandas.DataFrame): Dataframe containing images from a single scan volume indexed by ImageId.
            name (str): The name of the group identified by scan volume ID.
            scan_height (int): Image height in pixels.
            scan_width (int): Image width in pixels.
            scan_spacing (float): The spacing between scan centers in pixels.

        Returns:
            (matplotlib.figure.Figure, array(AxesSubplot)): Tuple of figure handle and array of subplot axes.
        """
        fig, ax = plt.subplots(1,2,figsize=[18,9],dpi=120)
        
        enface = self.get_enface_dt(grp,scan_height,scan_width,scan_spacing)
        ax[0].imshow(enface)
        ax[0].set_title(str(name) + ' DT')
        ax[0].set_aspect('equal')


        enface = self.get_enface_gt(grp,scan_height,scan_width,scan_spacing)      
        ax[1].imshow(enface)
        ax[1].set_title(str(name) + ' GT')
        ax[1].set_aspect('equal')
        return fig,ax                    

def Wilson_CI(p,n,z):
    if (p<0 or p>1 or n==0):
        if (p<0 or p>1):
            warnings.warn(f'The value of proportion {p} must be in the range [0,1]. Returning identity for CIs.')
        else:
            warnings.warn(f'The number of counts {n} must be above zero. Returning identity for CIs.')
        return (p,p)
    sym = z*(p*(1-p)/n + z*z/4/n/n)**.5
    asym = p + z*z/2/n
    fact = 1/(1+z*z/n)
    upper = fact*(asym+sym)
    lower = fact*(asym-sym)
    return (lower,upper)

class EvaluateClass(COCOEvaluator):  
    def __init__(self,dataset_name, output_dir,prob_thresh=0.5,iou_thresh = 0.1,evalsuper=True):
        super().__init__(dataset_name,tasks={'bbox','segm'},output_dir = output_dir)
        self.dataset_name = dataset_name
        self.mycoco=None #pycocotools.cocoEval instance
        self.cocoDt=None
        self.cocoGt=None
        self.evalsuper = evalsuper # if True, run COCOEvaluator.evaluate() when self.evaluate is run
        self.prob_thresh = prob_thresh #instance probabilty threshold for scalars (precision,recall,fpr for scans)
        self.iou_thresh = iou_thresh #iou threshold for defining precision,recall
        self.pr = None
        self.rc = None
        self.fpr = None
    def reset(self): 
        super().reset()
        self.mycoco=None
    def process(self, inputs, outputs):
        super().process(inputs,outputs)
    def evaluate(self):
        #with nostdout(): #suppress the coco summarize statment (the one with APs)
        if self.evalsuper:
            _ = super().evaluate() #this call populates coco_instances_results.json
        comm.synchronize()
        if not comm.is_main_process():
            return ()
        self.cocoGt = COCO(os.path.join(self._output_dir,self.dataset_name +'_coco_format.json')) #produced when super is initialized
        self.cocoDt = self.cocoGt.loadRes(os.path.join(self._output_dir,'coco_instances_results.json')) #load detector results
        self.mycoco = COCOeval(self.cocoGt,self.cocoDt,iouType ='segm')
        self.num_images = len(self.mycoco.params.imgIds)
        print('Calculated metrics for {} images'.format(self.num_images))
        self.mycoco.params.iouThrs = np.arange(.10,.6,.1)
        #mycoco.params.recThrs = [0,.5,.75,1]
        self.mycoco.params.maxDets = [100]
        #mycoco.params.imgIds=[imgId]
        self.mycoco.params.areaRng = [[0, 10000000000.0]]

        self.mycoco.evaluate()
        self.mycoco.accumulate()

        self.pr = self.mycoco.eval['precision'][:, #iouthresh
                         :, #recall level
                         0, #catagory
                         0, #area range
                         0] #max detections per image
        self.rc = self.mycoco.params.recThrs
        self.iou = self.mycoco.params.iouThrs
        self.scores = self.mycoco.eval['scores'][:,:,0,0,0] #unreliable if GT has no instances
        p,r = self.get_precision_recall()
        return p,r

    def plot_PRcurve(self,ax=None):
        if ax==None:
            fig, ax = plt.subplots(1,1)
        for i in range(len(self.iou)):
            ax.plot(self.rc,self.pr[i],label = '{:.2}'.format(self.iou[i]))
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('')
        ax.legend(title='IoU')
        
    def plot_recall_vs_prob(self):
        plt.figure()
        for i in range(len(self.iou)):
            plt.plot(self.rc,self.scores[i],label = '{:.2}'.format(self.iou[i]))
        plt.ylabel('Model probability')
        plt.xlabel('Recall')
        plt.legend(title='IoU')
            
    def get_precision_recall(self):
        iou_idx,rc_idx = self._find_iou_rc_inds()
        precision = self.pr[iou_idx,rc_idx]
        recall = self.rc[rc_idx]
        return precision,recall

    def _calculate_fpr_matrix(self):
        #FP rate, 1 RPD in image = FP
        if (self.scores.min()==-1) and (self.scores.max()==-1):
            print('WARNING: Scores for all iou thresholds and all recall levels are not defined. This can arise if ground truth annotations contain no instances. Leaving fpr matrix as None')
            self.fpr = None
            return

        fpr = np.zeros((len(self.iou),len(self.rc)))
        for i in range(len(self.iou)):
            for j,s in enumerate(self.scores[i]): #j -> recall level, s -> corresponding score
                ng = 0 #number of negative images
                fp = 0 #number of false positives images
                for el in self.mycoco.evalImgs:
                    if el is None:#no predictions, no gts
                        ng=ng+1
                    elif len(el['gtIds'])==0:# some predictions and no gts
                        ng=ng+1
                        if (np.array(el['dtScores']) >s).sum() > 0: #if at least one score over threshold for recall level
                            fp=fp+1 #count as FP
                    else:
                        continue
                fpr[i,j] = fp/ng
        self.fpr = fpr 

    def _calculate_fpr(self):
        print('Using alternate calculation for fpr at instance score threshold of {}'.format(self.prob_thresh))
        ng = 0 #number of negative images
        fp = 0 #number of false positives images
        for el in self.mycoco.evalImgs:
            if el is None:#no predictions, no gts
                ng=ng+1
            elif len(el['gtIds'])==0:# some predictions and no gts
                ng=ng+1
                if (np.array(el['dtScores']) >self.prob_thresh).sum() > 0: #if at least one score over threshold for recall level
                    fp=fp+1 #count as FP
            else: #gt has instances
                continue
        return fp/(ng+1e-5)

    def _find_iou_rc_inds(self):
        try:
            iou_idx = np.argwhere(self.iou==self.iou_thresh)[0][0] #first instance of
        except IndexError:
            print('iou threshold {} not found in mycoco.params.iouThrs {}'.format(self.iou_thresh,self.mycoco.params.iouThrs))
            exit(1)
        #test above for out of bounds
        inds = np.argwhere(self.scores[iou_idx]>=self.prob_thresh)
        if len(inds)>0:
            rc_idx = inds[-1][0] #get recall index corresponding to prob_thresh
        else:
            rc_idx = 0
        return iou_idx,rc_idx

    def get_fpr(self):
        
        if self.fpr is None:
            self._calculate_fpr_matrix()

        if self.fpr is not None: 
            iou_idx,rc_idx = self._find_iou_rc_inds()
            fpr = self.fpr[iou_idx,rc_idx]
        elif len(self.mycoco.cocoGt.anns)==0:
            fpr = self._calculate_fpr()
        else:
            fpr=-1
        return fpr
    
    def summarize_scalars(self): #for pretty printing 
        p,r = self.get_precision_recall()
        f1 = 2*(p*r)/(p + r)
        fpr = self.get_fpr()

        #Confidence intervals
        z=1.96 #95% Gaussian
        #instance count 
        inst_cnt = self.count_instances()
        n_r = inst_cnt['gt_instances']
        n_p = inst_cnt['dt_instances']
        n_fpr = inst_cnt['gt_neg_scans']
        
        def stat_CI(p,n,z):
            return z*np.sqrt(p*(1-p)/n)

        r_ci = Wilson_CI(r,n_r,z) 
        p_ci = Wilson_CI(p,n_p,z)
        fpr_ci = Wilson_CI(fpr,n_fpr,z)

        #propogate errors for f1
        int_r = stat_CI(r,n_r,z)
        int_p = stat_CI(p,n_p,z)
        int_f1 =(f1)*np.sqrt(int_r**2 * (1/r - 1/(p+r))**2 + int_p**2 * (1/p - 1/(p+r))**2)
        f1_ci = (f1-int_f1,f1+int_f1)

        dd = dict(dataset = self.dataset_name, precision = float(p),precision_ci = p_ci,recall=float(r), recall_ci = r_ci,f1 = float(f1),f1_ci = f1_ci, fpr=float(fpr),fpr_ci = fpr_ci, iou=self.iou_thresh,probability=self.prob_thresh)
        return dd

    def count_instances(self):
        gt_inst = 0
        dt_inst = 0
        gt_neg_scans = 0
        for key,val in self.cocoGt.imgs.items():
            imgid = val['id']
            #Gt instances
            annIdsGt = self.cocoGt.getAnnIds([imgid])
            annsGt = self.cocoGt.loadAnns(annIdsGt)
            gt_inst+=len(annsGt)
            if len(annsGt)==0:
                gt_neg_scans+=1

            #Dt instances
            annIdsDt = self.cocoDt.getAnnIds([imgid])
            annsDt = self.cocoDt.loadAnns(annIdsDt)
            annsDt = [ann for ann in annsDt if ann['score']>self.prob_thresh]
            dt_inst+=len(annsDt)

        return dict(gt_instances=gt_inst,dt_instances=dt_inst,gt_neg_scans = gt_neg_scans)


from sklearn.metrics import precision_recall_curve,average_precision_score
class CreatePlotsRPD():
    def __init__(self,dfimg):
        self.dfimg = dfimg
        self.dfvol = self.dfimg.groupby(['volID'])[['gt_instances','gt_pxs','gt_xpxs','dt_instances','dt_pxs','dt_xpxs']].sum()
        
    @classmethod
    def initfromcoco(cls,mycoco,prob_thresh):
        df = pd.DataFrame(index = mycoco.cocoGt.imgs.keys(), columns=['gt_instances','gt_pxs','gt_xpxs','dt_instances','dt_pxs','dt_xpxs'],dtype=np.uint64)

        for key,val in mycoco.cocoGt.imgs.items():
            imgid = val['id']
            #Gt instances
            annIdsGt = mycoco.cocoGt.getAnnIds([imgid])
            annsGt = mycoco.cocoGt.loadAnns(annIdsGt)
            instGt = [mycoco.cocoGt.annToMask(ann).sum() for ann in annsGt]
            xprojGt = [(mycoco.cocoGt.annToMask(ann).sum(axis=0)>0).astype('uint8').sum() for ann in annsGt] 
            #Dt instances
            annIdsDt = mycoco.cocoDt.getAnnIds([imgid])
            annsDt = mycoco.cocoDt.loadAnns(annIdsDt)
            annsDt = [ann for ann in annsDt if ann['score']>prob_thresh]
            instDt = [mycoco.cocoDt.annToMask(ann).sum() for ann in annsDt]
            xprojDt = [(mycoco.cocoDt.annToMask(ann).sum(axis=0)>0).astype('uint8').sum() for ann in annsDt]
                
            dat = [len(instGt),np.array(instGt).sum(),np.array(xprojGt).sum(),len(instDt),np.array(instDt).sum(),np.array(xprojDt).sum()]
            df.loc[key] = dat
    
        newdf = pd.DataFrame([idx.rsplit('.',1)[0].rsplit('_',1) for idx in df.index],columns=['volID','scan'],index = df.index)
        df = df.merge(newdf,how='inner',left_index=True,right_index=True)
        return cls(df)
    
    @classmethod
    def initfromcsv(cls,fname):
        df = pd.read_csv(fname)
        return cls(df)
           
    def get_max_limits(self,df):
        max_inst=np.max([df.gt_instances.max(),df.dt_instances.max()])
        max_xpxs = np.max([df.gt_xpxs.max(),df.dt_xpxs.max()])
        max_pxs = np.max([df.gt_pxs.max(),df.dt_pxs.max()])
#         print('Max instances:',max_inst)
#         print('Max xpxs:',max_xpxs)
#         print('Max pxs:',max_pxs)
        return max_inst,max_xpxs,max_pxs

    def vol_level_prc(self,df,gt_thresh=5,ax=None):

        prc = precision_recall_curve(df.gt_instances>=gt_thresh,df.dt_instances)
        if ax==None:
            fig,ax = plt.subplots(1,1)
        ax.plot(prc[1],prc[0])
        ax.set_xlabel('RPD Volume Recall')
        ax.set_ylabel('RPD Volume Precision')


        ap = average_precision_score(df.gt_instances>=gt_thresh,df.dt_instances)
        return ap,prc

    def plot_img_level_instance_thresholding(self,df,inst):

        rc = np.zeros((len(inst),))
        pr = np.zeros((len(inst),))
        fpr = np.zeros((len(inst),))

        fig, ax = plt.subplots(1,3,figsize = [15,5])
        for i,dt_thresh in enumerate(inst):
            gt = df.gt_instances>dt_thresh
            dt = df.dt_instances>dt_thresh
            rc[i] = (gt&dt).sum()/gt.sum()
            pr[i] = (gt&dt).sum()/dt.sum()
            fpr[i] = ((~gt)&(dt)).sum()/((~gt).sum())

        ax[1].plot(inst,pr)
        ax[1].set_ylim(0.45,1.01)
        ax[1].set_xlabel('instance threshold')
        ax[1].set_ylabel('Precision')


        ax[0].plot(inst,rc)
        ax[0].set_ylim(0.45,1.01)
        ax[0].set_ylabel('Recall')
        ax[0].set_xlabel('instance threshold')


        ax[2].plot(inst,fpr)
        ax[2].set_ylim(0,0.80)
        ax[2].set_xlabel('instance threshold')
        ax[2].set_ylabel('FPR')

        plt.tight_layout()
        return pr,rc,fpr

    def plot_img_level_instance_thresholding2(self,df,inst,gt_thresh,plot=True):

        #should be using global function
        # def Wilson_CI(p,n,z):

        #     sym = z*(p*(1-p)/n + z*z/4/n/n)**.5
        #     asym = p + z*z/2/n
        #     fact = 1/(1+z*z/n)
        #     upper = fact*(asym+sym)
        #     lower = fact*(asym-sym)
        #     return (lower,upper)

        rc = np.zeros((len(inst),))
        pr = np.zeros((len(inst),))
        fpr = np.zeros((len(inst),))
        rc_ci = np.zeros((len(inst),2))
        pr_ci = np.zeros((len(inst),2))
        fpr_ci = np.zeros((len(inst),2))

        
        for i,dt_thresh in enumerate(inst):
            gt = df.gt_instances>=gt_thresh
            dt = df.dt_instances>=dt_thresh
            rc[i] = (gt&dt).sum()/gt.sum()
            pr[i] = (gt&dt).sum()/dt.sum()
            fpr[i] = ((~gt)&(dt)).sum()/((~gt).sum())
            rc_ci[i,:] = Wilson_CI(rc[i],gt.sum(),1.96)
            pr_ci[i,:]= Wilson_CI(pr[i],dt.sum(),1.96)
            fpr_ci[i,:] = Wilson_CI(fpr[i],((~gt).sum()),1.96)

        if plot:
            fig, ax = plt.subplots(1,3,figsize = [15,5])
            # ax[0].plot(rc,pr)
            # ax[0].set_xlabel('Recall')
            # ax[0].set_ylabel('Precision')

            ax[1].plot(inst,pr)
            ax[1].fill_between(inst,pr_ci[:,0],pr_ci[:,1],alpha=.25)
            #ax[1].set_ylim(0.45,1.01)
            ax[1].set_xlabel('instance threshold')
            ax[1].set_ylabel('Precision')


            ax[0].plot(inst,rc)
            ax[0].fill_between(inst,rc_ci[:,0],rc_ci[:,1],alpha=.25)
            #ax[0].set_ylim(0.45,1.01)
            ax[0].set_ylabel('Recall')
            ax[0].set_xlabel('instance threshold')


            ax[2].plot(inst,fpr)
            ax[2].fill_between(inst,fpr_ci[:,0],fpr_ci[:,1],alpha=.25)
            #ax[2].set_ylim(0,0.80)
            ax[2].set_xlabel('instance threshold')
            ax[2].set_ylabel('FPR')

            plt.tight_layout()
        return dict(precision=pr,precision_ci = pr_ci,recall=rc,recall_ci = rc_ci, fpr=fpr,fpr_ci = fpr_ci)

    def gt_vs_dt_instances(self,ax=None):
        df = self.dfimg
        max_inst,max_xpxs,max_pxs = self.get_max_limits(df)
        idx = (df.gt_instances>0)&(df.dt_instances>0)
        
        if ax==None:
            fig = plt.figure(dpi=100)
            ax = fig.add_subplot(111)

        y = df[idx].groupby('gt_instances')['dt_instances'].mean()
        yerr = df[idx].groupby('gt_instances')['dt_instances'].std()
        ax.errorbar(y.index,y.values,yerr.values,fmt='*')
        plt.plot([0,max_inst],[0,max_inst],alpha=.5)
        plt.xlim(0,max_inst+1)
        plt.ylim(0,max_inst+1)
        ax.set_aspect(1)
        plt.xlabel('gt_instances')
        plt.ylabel('dt_instances')
        plt.tight_layout()
        return ax

    def gt_vs_dt_instances_boxplot(self,ax=None):
        df = self.dfimg
        max_inst,max_xpxs,max_pxs = self.get_max_limits(df) 
        max_inst = int(max_inst)
        if ax==None:
            fig = plt.figure(dpi=100)
            ax = fig.add_subplot(111)
        
       


        ax.plot([0,max_inst+1],[0,max_inst+1],alpha=.5)
        x = df['gt_instances'].values.astype(int)
        y = df['dt_instances'].values.astype(int)
        sns.boxplot(x,y, ax=ax,width=.5)
        ax.set_xbound(0,max_inst+1)
        ax.set_ybound(0,max_inst+1)        
        ax.set_aspect('equal')
        

        ax.set_title('')
        ax.set_xlabel('gt_instances')
        ax.set_ylabel('dt_instances')

        import matplotlib.ticker as pltticker
        loc = pltticker.MultipleLocator(base=2.0)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)


        return ax



    def gt_vs_dt_xpxs(self):
        df = self.dfimg
        max_inst,max_xpxs,max_pxs = self.get_max_limits(df)
        idx = (df.gt_instances>0)&(df.dt_instances>0)
        dfsub = df[idx]
        
        fig1 = plt.figure(figsize = [10,10],dpi=100)
        ax = fig1.add_subplot(111)
        sc = ax.scatter(dfsub['gt_xpxs'],dfsub['dt_xpxs'],c =dfsub['gt_instances'] ,cmap='viridis')
        ax.set_aspect(1)
        #ax = dfsub.plot(kind = 'scatter',x=,y=,c='gt_instances')
        plt.plot([0,max_xpxs],[0,max_xpxs],alpha=.5)
        plt.xlim(0,max_xpxs)
        plt.ylim(0,max_xpxs)
        plt.xlabel('gt_xpxs')
        plt.ylabel('dt_xpxs')
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel('gt_instances')
        plt.tight_layout()
        
        fig2 = plt.figure(figsize = [10,10],dpi=100)
        ax = fig2.add_subplot(111)
        sc = ax.scatter(dfsub['gt_xpxs'],dfsub['gt_xpxs']-dfsub['dt_xpxs'],c =dfsub['gt_instances'] ,cmap='viridis')
        #ax = dfsub.plot(kind = 'scatter',x=,y=,c='gt_instances')
        plt.plot([0,max_xpxs],[0,0],alpha=.5)
        plt.xlabel('gt_xpxs')
        plt.ylabel('gt_xpxs-dt_xpxs')
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel('gt_instances')
        plt.tight_layout()
        
        fig3 = plt.figure(dpi=100)
        plt.hist(dfsub['gt_xpxs']-dfsub['dt_xpxs'])
        plt.xlabel('gt_xpxs - dt_xpxs')
        plt.ylabel('B-scans')
        
        return fig1,fig2,fig3
    
    def gt_vs_dt_xpxs_mu(self):
        df = self.dfimg
        max_inst,max_xpxs,max_pxs = self.get_max_limits(df)
        idx = (df.gt_instances>0)&(df.dt_instances>0)
        dfsub = df[idx]
        
        from scipy import stats
        mu_dt,bins,bnum = stats.binned_statistic(dfsub['gt_xpxs'],dfsub['dt_xpxs'],statistic = 'mean',bins=10)
        std_dt,_,_ = stats.binned_statistic(dfsub['gt_xpxs'],dfsub['dt_xpxs'],statistic = 'std',bins = bins)
        mu_gt,_,_ = stats.binned_statistic(dfsub['gt_xpxs'],dfsub['gt_xpxs'],statistic='mean',bins=bins)
        std_gt,_,_ = stats.binned_statistic(dfsub['gt_xpxs'],dfsub['gt_xpxs'],statistic = 'std',bins = bins)
        fig = plt.figure(dpi=100)
        plt.errorbar(mu_gt,mu_dt,yerr = std_dt,xerr=std_gt,fmt='*')
        plt.xlabel('gt_xpxs')
        plt.ylabel('dt_xpxs')
        plt.plot([0,max_xpxs],[0,max_xpxs],alpha=.5)
        plt.xlim(0,max_xpxs)
        plt.ylim(0,max_xpxs)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        return fig

    def gt_dt_FP_FN_count(self):
        df = self.dfimg
        fig,ax =plt.subplots(1,2,figsize=[10,5])

        idx = (df.gt_instances==0)&(df.dt_instances>0)
        ax[0].hist(df[idx]['dt_instances'],bins = range(1,10))
        ax[0].set_xlabel('dt instances')
        ax[0].set_ylabel('B-scans')
        ax[0].set_title('FP dt instance count per B-scan')

        idx = (df.gt_instances>0)&(df.dt_instances==0)
        ax[1].hist(df[idx]['gt_instances'],bins = range(1,10))
        ax[1].set_xlabel('gt instances')
        ax[1].set_ylabel('B-scans')
        ax[1].set_title('FN gt instance count per B-scan')

        plt.tight_layout()
        return fig
    
    def avg_inst_size(self):
        df = self.dfimg
        max_inst,max_xpxs,max_pxs = self.get_max_limits(df)
        idx = (df.gt_instances>0)&(df.dt_instances>0)
        dfsub = df[idx]
        
        fig = plt.figure(figsize=[10,5])
        plt.subplot(121)
        bins = np.arange(0,120,10)
        ax = (dfsub.gt_xpxs/dfsub.gt_instances).hist(bins = bins,alpha=.5,label='gt')
        ax = (dfsub.dt_xpxs/dfsub.dt_instances).hist(bins=bins,alpha=.5,label='dt')
        ax.set_xlabel('xpxs')
        ax.set_ylabel('B-scans')
        ax.set_title('Average size of instance')
        ax.legend()

        plt.subplot(122)
        bins = np.arange(0,600,40)
        ax = (dfsub.gt_pxs/dfsub.gt_instances).hist(bins=bins,alpha=.5,label='gt')
        ax = (dfsub.dt_pxs/dfsub.dt_instances).hist(bins=bins,alpha=.5,label='dt')
        ax.set_xlabel('pxs')
        ax.set_ylabel('B-scans')
        ax.set_title('Average size of instance')
        ax.legend()

        plt.tight_layout()
        return fig
    

