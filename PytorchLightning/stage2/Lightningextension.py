import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import data.transforms_bbox as Tr
from data.voc import VOC_box
from models.ClsNet import Labeler, pad_for_grid
from utils.densecrf import DENSE_CRF
from PIL import Image

def my_collate(batch):
    '''
    This is to assign a batch-wise index for each box.
    '''
    sample = {}
    img = []
    bboxes = []
    bg_mask = []
    batchID_of_box = []
    filenames=[]
    items=[]
    for batch_id, item in enumerate(batch):
        img.append(item[0])
        bboxes.append(item[1]) 
        bg_mask.append(item[2])
        filenames.append(item[3])
        items.append(item)
        for _ in range(len(item[1])):
            batchID_of_box += [batch_id]
    sample["img"] = torch.stack(img, dim=0)
    sample["bboxes"] = torch.cat(bboxes, dim=0)
    sample["bg_mask"] = torch.stack(bg_mask, dim=0)[:,None]
    sample["batchID_of_box"] = torch.tensor(batchID_of_box, dtype=torch.long)
    sample["filenames"]=filenames
    sample["item"]=items
    return sample

class VOCDataModule(pl.LightningDataModule):
    def __init__( cfg):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # Check whether to save hparams 
        # self.save_hyperparameters(logger=False)      
        # data transformations
        self.transforms= Tr.Normalize_Caffe()
        # self.dims= Should be the size of each image
        self.dataset=None
        self.cfg=cfg 
    @ property
    def num_classes(self) -> int:
        return self.cfg.DATA.NUM_CLASSES
    def prepare_data(self):
        # Not needed
        pass 
    def setup(self):
        # Not needed
        pass 
    def train_dataloader(self):
        return None
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=1,collate_fn=my_collate)
    def test_dataloader(self):
        return None

class LightningModel(pl.LightningModule):
    def __init__(self,cfg):
        super().__init__()
        self.model=Labeler(cfg.DATA.NUM_CLASSES, cfg.MODEL.ROI_SIZE, cfg.MODEL.GRID_SIZE)
        self.cfg=cfg
        self.classifier_weights=torch.clone(self.model.classifier.weight.data)
        # load checkpoint 
        # need to see whether the function also works on .pt files
        self.load_weights(f"./weights/{cfg.MODEL.WEIGHTS}")  # Just loading pre-trained weights
    
    def DENSE_CRF(self):
        bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std = self.cfg.MODEL.DCRF
        return DENSE_CRF(bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
    
    def CHECK_SAVE_PSEUDO_LABLES(self):
        save_paths = []
        if self.cfg.SAVE_PSEUDO_LABLES:
         folder_name = os.path.join(self.cfg.DATA.ROOT, self.cfg.NAME)
         os.mkdir(folder_name)
        for txt in ("Y_crf", "Y_ret"):
            sub_folder = folder_name + f"/{txt}"
            os.mkdir(sub_folder)
            save_paths += [os.path.join(sub_folder, "{}.png")]
        return save_paths
    
    def val_step(self, batch, batch_idx): 
        # check for validation/testing
        sample=val_batch  
        self.common_step(sample)  
    
    def common_step(self,sample):
        self.get_features(sample)
        self.get_bg_attn(sample)
        self.get_unary(sample)
        Y_crf,Y_ret=self.get_Y_ret()
        paths=self.CHECK_SAVE_PSEUDO_LABLES():
        if paths:
            for pseudo, save_path in zip([Y_crf, Y_ret],paths):
                Image.fromarray(pseudo).save(save_path.format(sample["filenames"]))
    
    def getlabels(self,sample):
        fn = sample["filenames"]
        self.rgb_img = np.array(Image.open(sample["item"].img_path.format(fn)))
        sample["bboxes"]=sample["bboxes"][0]
        sample["bg_mask"]=sample["bg_mask"][None]
        self.img_H, self.img_W = sample["img"].shape[-2:]
        norm_H, norm_W = (self.img_H-1)/2, (self.img_W-1)/2
        sample["bboxes"][:,[0,2]] = sample["bboxes"][:,[0,2]]*norm_W + norm_W
        sample["bboxes"][:,[1,3]] = sample["bboxes"][:,[1,3]]*norm_H + norm_H
        sample["bboxes"] = sample["bboxes"].long()
        self.gt_labels =sample["bboxes"][:,4].unique()
    
    def get_features(self,sample):
        self.features = self.model.get_features(sample["img"])
        self.features = F.interpolate(self.features,sample["img"].shape[-2:], mode='bilinear', align_corners=True)
        self.normed_f = F.normalize(self.features)
        self.padded_features = pad_for_grid(self.features, self.cfg.MODEL.GRID_SIZE)
    
    def get_bg_attn(self,sample):
        self.padded_bg_mask = pad_for_grid(sample["bg_mask"], self.cfg.MODEL.GRID_SIZE)
        self.grid_bg, self.valid_gridIDs = self.model.get_grid_bg_and_IDs(self.padded_bg_mask, self.cfg.MODEL.GRID_SIZE)
        self.bg_protos = self.model.get_bg_prototypes(self.padded_features, self.padded_bg_mask, self.grid_bg, self.cfg.MODEL.GRID_SIZE)
        self.bg_protos = self.bg_protos[0,valid_gridIDs] # (1,GS**2,dims,1,1) --> (len(valid_gridIDs),dims,1,1)
        self.normed_bg_p = F.normalize(self.bg_protos)
        self.bg_attns = F.relu(torch.sum(self.normed_bg_p*self.normed_f, dim=1))
        self.bg_attn = torch.mean(self.bg_attns, dim=0, keepdim=True) # (len(valid_gridIDs),H,W) --> (1,H,W)
        self.bg_attn[self.bg_attn < self.cfg.MODEL.BG_THRESHOLD * self.bg_attn.max()] = 0
    
    def get_Bg_unary(self,sample):
        self.Bg_unary = torch.clone(sample["bg_mask"][0]) # (1,H,W)
        self.region_inside_bboxes = self.Bg_unary[0]==0 # (H,W)
        self.Bg_unary[:,region_inside_bboxes] = self.bg_attn[:,region_inside_bboxes].detach().cpu()
    
    def get_Fg_unary(self,sample):
        self.Fg_unary = []
        self.getlabels(sample)
        for uni_cls in self.gt_labels:
            w_c = self.classifier_weights[uni_cls][None]
            raw_cam = F.relu(torch.sum(w_c*self.features, dim=1)) # (1,H,W)
            normed_cam = torch.zeros_like(raw_cam)
            for wmin,hmin,wmax,hmax,_ in sample["bboxes"][sample["bboxes"][:,4]==uni_cls]:
                denom = raw_cam[:,hmin:hmax,wmin:wmax].max() + 1e-12
                normed_cam[:,hmin:hmax,wmin:wmax] = raw_cam[:,hmin:hmax,wmin:wmax] / denom
            self.Fg_unary += [normed_cam]
        self.Fg_unary = torch.cat(Fg_unary, dim=0).detach().cpu()
    
    def get_unary(self,sample):
        self.get_Bg_unary(sample)
        self.get_Fg_unary(sample)
        self.unary = torch.cat((self.Bg_unary,self.Fg_unary), dim=0)
        self.unary[:,region_inside_bboxes] = torch.softmax(self.unary[:,region_inside_bboxes], dim=0)
        self.refined_unary = self.DENSE_CRF.inference(self.rgb_img, unary.numpy())
        
        for idx_cls, uni_cls in enumerate(self.gt_labels,1):
            mask = np.zeros((self.img_H,self.img_W))
            for wmin,hmin,wmax,hmax,_ in sample["bboxes"][sample["bboxes"][:,4]==uni_cls]:
                mask[hmin:hmax,wmin:wmax] = 1
            self.refined_unary[idx_cls] *= mask
    
    def get_Y_crf(self):
        tmp_mask = self.refined_unary.argmax(0)
        Y_crf = np.zeros_like(tmp_mask, dtype=np.uint8)
        for idx_cls, uni_cls in enumerate(self.gt_labels,1):
            Y_crf[tmp_mask==idx_cls] = uni_cls
        Y_crf[tmp_mask==0] = 0
        return Y_crf
    
    def get_Y_ret(self):
        Y_crf=self.get_Y_crf()
        tmp_Y_crf = torch.from_numpy(Y_crf)
        gt_labels_with_Bg = [0] +self.gt_labels.tolist()

        corr_maps = []
        for uni_cls in gt_labels_with_Bg:
            indices = tmp_Y_crf==uni_cls
            if indices.sum():
                normed_p = F.normalize(self.features[...,indices].mean(dim=-1))   # (1,dims)
                corr = F.relu((self.normed_f*normed_p[...,None,None]).sum(dim=1)) # (1,H,W)
            else:
                normed_w = F.normalize(self.classifier_weights[uni_cls][None])
                corr = F.relu((self.normed_f*normed_w).sum(dim=1)) # (1,H,W)
            corr_maps.append(corr)
        corr_maps = torch.cat(corr_maps) # (1+len(gt_labels),H,W)
        
        for idx_cls, uni_cls in enumerate(gt_labels_with_Bg):
            if uni_cls == 0:
                    corr_maps[idx_cls, ~self.region_inside_bboxes] = 1
            else:
                mask = torch.zeros(self.img_H,self.img_W).type_as(corr_maps)
                for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                    mask[hmin:hmax,wmin:wmax] = 1
                corr_maps[idx_cls] *= mask
        
        tmp_mask = corr_maps.argmax(0).detach().cpu().numpy()
        Y_ret = np.zeros_like(tmp_mask, dtype=np.uint8)
        for idx_cls, uni_cls in enumerate(self.gt_labels,1):
            Y_ret[tmp_mask==idx_cls] = uni_cls
        Y_ret[tmp_mask==0] = 0
        return Y_crf,Y_ret
    
    def load_weights(self,path):
        self.model.load_from_checkpoint(path)



    








    
    