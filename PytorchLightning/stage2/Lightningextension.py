import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import data.transforms_bbox as Tr
from data.voc import VOC_box
from models.ClsNet import Labeler
from utils.densecrf import DENSE_CRF
import os

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
    @ property
    def num_classes(self) -> int:
        return cfg.DATA.NUM_CLASSES
    def prepare_data(self):
        # Not needed
        pass 
    def setup(self):
        if cfg.DATA.MODE == "train":
            txt_name = "train_aug.txt"
        if cfg.DATA.MODE == "val":
            txt_name = "val.txt"
        self.dataset = VOC_box(cfg, self.transforms)
    def train_dataloader(self):
        if cfg.DATA.MODE == "train":
            return DataLoader(self.dataset, batch_size=1)
        return None
    def val_dataloader(self):
        if cfg.DATA.MODE == "val":
            return DataLoader(self.dataset, batch_size=1)
        return None
    def test_dataloader(self):
        ''' NEED TO CHECK THIS'''
        if cfg.DATA.MODE == "val":
            return DataLoader(self.dataset, batch_size=1)
        return None

class LabelerLitModel(pl.LightningModule):
    def __init__(self,cfg):
        super().__init__()
        self.model=Labeler(cfg.DATA.NUM_CLASSES, cfg.MODEL.ROI_SIZE, cfg.MODEL.GRID_SIZE)
        self.cfg=cfg
    
    def DENSE_CRF(self):
        bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std = self.cfg.MODEL.DCRF
        return DENSE_CRF(bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
    
    def CHECK_SAVE_PSEUDO_LABLES(self):
        save_paths = []
        if self.cfg.SAVE_PSEUDO_LABLES:
        folder_name = os.path.join(cfg.DATA.ROOT, cfg.NAME)
        os.mkdir(folder_name)
        for txt in ("Y_crf", "Y_ret"):
            sub_folder = folder_name + f"/{txt}"
            os.mkdir(sub_folder)
            save_paths += [os.path.join(sub_folder, "{}.png")]
        return save_paths
    
    