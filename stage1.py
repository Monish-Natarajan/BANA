import pytorch-lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import os
import sys
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import data.transforms_bbox as Tr
from data.voc import VOC_box
from configs.defaults import _C
from models.ClsNet import Labeler

# logger = logging.getLogger("stage1")


def my_collate(batch):
    '''
    This is to assign a batch-wise index for each box.
    '''
    sample = {}
    img = []
    bboxes = []
    bg_mask = []
    batchID_of_box = []
    for batch_id, item in enumerate(batch):
        img.append(item[0])
        bboxes.append(item[1]) 
        bg_mask.append(item[2])
        for _ in range(len(item[1])):
            batchID_of_box += [batch_id]
    sample["img"] = torch.stack(img, dim=0)
    sample["bboxes"] = torch.cat(bboxes, dim=0)
    sample["bg_mask"] = torch.stack(bg_mask, dim=0)[:,None]
    sample["batchID_of_box"] = torch.tensor(batchID_of_box, dtype=torch.long)
    return sample

wandb_logger = WandbLogger(project='BANA', # group runs in "BANA" project
                           log_model='all') # log all new checkpoints during training

class VOCDataModule(pl.LightningDataModule):
    def __init__( cfg):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # Check whether to save hparams 
        self.save_hyperparameters(logger=False)      
        # data transformations
        self.transforms= Tr.Compose([
        Tr.RandomScale(0.5, 1.5),
        Tr.ResizeRandomCrop(cfg.DATA.CROP_SIZE), 
        Tr.RandomHFlip(0.5), 
        Tr.ColorJitter(0.5,0.5,0.5,0),
        Tr.Normalize_Caffe(),
        ])
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
            return DataLoader(self.dataset, batch_size=cfg.DATA.BATCH_SIZE,collate_fn=my_collate,shuffle=True,num_workers=4,pin_memory=True,drop_last=True )
    def val_dataloader(self):
        if cfg.DATA.MODE == "val":
            return DataLoader(self.dataset, batch_size=cfg.DATA.BATCH_SIZE,collate_fn=my_collate,shuffle=True,num_workers=4,pin_memory=True,drop_last=True )
    def test_dataloader(self):
        ''' NEED TO CHECK THIS'''
        if cfg.DATA.MODE == "val":
            return DataLoader(self.dataset, batch_size=cfg.DATA.BATCH_SIZE,collate_fn=my_collate,shuffle=True,num_workers=4,pin_memory=True,drop_last=True )

class LabelerLitModel(pl.LightningModule):
    def __init__(self,cfg):
        super().__init__()
        self.model=Labeler(cfg.DATA.NUM_CLASSES, cfg.MODEL.ROI_SIZE, cfg.MODEL.GRID_SIZE)
        self.cfg=cfg
        self.params = self.model.get_params()
        self.criterion = nn.CrossEntropyLoss()
        self.interval_verbose = self.cfg.SOLVER.MAX_ITER // 40
        self.val_losses = []                               # can replace by floats
        self.avgval_losses = []
        self.train_losses = []
        self.avgtrain_losses = []

    def training_step(self, batch, batch_idx):
        if self.cfg.DATA.MODE == "train":
         sample=train_batch                      # Need to check whether validation and training to be done at the same time
         loss = common_step(sample)
         self.train_losses.append(loss.item())
         result=pl.TrainResult(loss)
         return result

    def validation_step(self, batch, batch_idx):
        if self.cfg.DATA.MODE == "val":
             #to do... right now kept same as train
             sample=val_batch                      # Need to check whether validation and training to be done at the same time
             loss = common_step(sample)
             self.val_losses.append(loss.item())
             result=pl.EvalResult(loss)
             return result

    def common_step(sample):
        img = sample["img"]
        bboxes = sample["bboxes"]
        bg_mask = sample["bg_mask"]
        batchID_of_box = sample["batchID_of_box"]
        ind_valid_bg_mask = bg_mask.mean(dim=(1,2,3)) > 0.125 # This is because VGG16 has output stride of 8.
        logits = self.model(img, bboxes, batchID_of_box, bg_mask, ind_valid_bg_mask)
        logits = logits[...,0,0]
        fg_t = bboxes[:,-1][:,None].expand(bboxes.shape[0], np.prod(self.cfg.MODEL.ROI_SIZE))
        fg_t = fg_t.flatten().long()
        target = torch.zeros(logits.shape[0], dtype=torch.long)
        target[:fg_t.shape[0]] = fg_t
        loss = self.criterion(logits, target)
        return loss 

    def validation_epoch_end(self, outputs):
        self.avgval_losses.append(sum(self.val_losses) / len(self.val_losses))
        self.val_losses=[]
        if self.current_epoch()+1 % self.interval_verbose ==0:
            # log
            self.avgval_losses=[]

    def training_epoch_end(self, outputs):
        self.avgtrain_losses.append(sum(self.val_losses) / len(self.train_losses))
        self.train_losses=[]
        if self.current_epoch()+1 % self.interval_verbose ==0:
            # log
            self.avgtrain_losses=[] 
    
    def test_step(self, batch, batch_idx):
        if self.cfg.DATA.MODE == "val":
             #to do... right now kept same as train
             sample=val_batch                      # Need to check this
             loss = common_step(sample)
             result=pl.EvalResult(loss)
             # ADD LOGGER-
             return result
    
    def test_epoch_end(self, outputs):
        # to do
        pass

    def configure_optimizers(self):
        lr = self.cfg.SOLVER.LR
        wd = self.cfg.SOLVER.WEIGHT_DECAY
        optimizer = optim.SGD(
        [{"params":self.params[0], "lr":lr,    "weight_decay":wd},
         {"params":self.params[1], "lr":2*lr,  "weight_decay":0 },
         {"params":self.params[2], "lr":10*lr, "weight_decay":wd},
         {"params":self.params[3], "lr":20*lr, "weight_decay":0 }], 
        momentum=self.cfg.SOLVER.MOMENTUM
        )
        lr_scheduler = {
        "scheduler": optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.SOLVER.MILESTONES, gamma=0.1),
        # 'epoch' updates the scheduler on epoch end whereas 'step' updates it after a optimizer update.
        "interval": "step",
        "frequency": 1,
        # Metric to to monitor for schedulers like `ReduceLROnPlateau`
        "monitor": "train_loss" if self.cfg.DATA.MODE == "train" else "val_loss",
        "strict": True,
        "name": None,
        }
        return { "optimizer": optimizer,"lr_scheduler":lr_scheduler }
        

def main(cfg):
    # logger.setLevel(logging.DEBUG)
    # logger.propagate = False
    # formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    # ch = logging.StreamHandler(stream=sys.stdout)
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(formatter)
    # logger.addHandler(ch)
    # fh = logging.FileHandler(f"./logs/{cfg.NAME}.txt")
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.info(" ".join(["\n{}: {}".format(k, v) for k,v in cfg.items()]))
    
    wandb_logger.watch(model)

    if cfg.SEED:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)

    # tr_transforms = Tr.Compose([
    #     Tr.RandomScale(0.5, 1.5),
    #     Tr.ResizeRandomCrop(cfg.DATA.CROP_SIZE), 
    #     Tr.RandomHFlip(0.5), 
    #     Tr.ColorJitter(0.5,0.5,0.5,0),
    #     Tr.Normalize_Caffe(),
    # ])
    # trainset = VOC_box(cfg, tr_transforms)
    # train_loader = DataLoader(trainset, batch_size=cfg.DATA.BATCH_SIZE, collate_fn=my_collate, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    # model = Labeler(cfg.DATA.NUM_CLASSES, cfg.MODEL.ROI_SIZE, cfg.MODEL.GRID_SIZE).cuda()
    model=LabelerLitModel(cfg)
    trainer = Trainer()
    # 
    model.backbone=model.backbone.load_from_checkpoint(f"./weights/{cfg.MODEL.WEIGHTS}")
    model.backbone.load_state_dict(torch.load(f"./weights/{cfg.MODEL.WEIGHTS}"), strict=False)
    
    # params = model.get_params()
    # lr = cfg.SOLVER.LR
    # wd = cfg.SOLVER.WEIGHT_DECAY
    # optimizer = optim.SGD(
    #     [{"params":params[0], "lr":lr,    "weight_decay":wd},
    #      {"params":params[1], "lr":2*lr,  "weight_decay":0 },
    #      {"params":params[2], "lr":10*lr, "weight_decay":wd},
    #      {"params":params[3], "lr":20*lr, "weight_decay":0 }], 
    #     momentum=cfg.SOLVER.MOMENTUM
    # )
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES, gamma=0.1)
    # criterion = nn.CrossEntropyLoss()
    
    # model.train()
    # iterator = iter(train_loader)                   
    # storages = {"CE": 0,}
    # interval_verbose = cfg.SOLVER.MAX_ITER // 40           # This is for max iterations to pass to log
    logger.info(f"START {cfg.NAME} -->")
    # for it in range(1, cfg.SOLVER.MAX_ITER+1):             # max epoch
        # try:
        #     sample = next(iterator)
        # except:
        #     iterator = iter(train_loader)
        #     sample = next(iterator)
        # img = sample["img"]
        # bboxes = sample["bboxes"]
        # bg_mask = sample["bg_mask"]
        # batchID_of_box = sample["batchID_of_box"]
        # ind_valid_bg_mask = bg_mask.mean(dim=(1,2,3)) > 0.125 # This is because VGG16 has output stride of 8.
        # logits = model(img.cuda(), bboxes, batchID_of_box, bg_mask.cuda(), ind_valid_bg_mask)
        # logits = logits[...,0,0]
        # fg_t = bboxes[:,-1][:,None].expand(bboxes.shape[0], np.prod(cfg.MODEL.ROI_SIZE))
        # fg_t = fg_t.flatten().long()
        # target = torch.zeros(logits.shape[0], dtype=torch.long)
        # target[:fg_t.shape[0]] = fg_t
        # loss = criterion(logits, target.cuda())
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # scheduler.step()
        # storages["CE"] += loss.item()
        # if it % interval_verbose == 0:
        #     for k in storages.keys(): storages[k] /= interval_verbose
        #     logger.info("{:3d}/{:3d}  Loss (CE): {:.4f}  lr: {}".format(it, cfg.SOLVER.MAX_ITER, storages["CE"], optimizer.param_groups[0]["lr"]))
        #     for k in storages.keys(): storages[k] = 0
    torch.save(model.state_dict(), f"./weights/{cfg.NAME}.pt")
    logger.info("--- SAVED ---")
    logger.info(f"END {cfg.NAME} -->")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    main(cfg)