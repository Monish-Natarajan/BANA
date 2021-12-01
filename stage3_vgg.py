import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import data.transforms_bbox as Tr
from data.voc import VOC_seg
from configs.defaults import _C

from models.SegNet import DeepLab_LargeFOV

def main(cfg):    
    if cfg.SEED:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)

    tr_transforms = Tr.Compose([
        Tr.RandomScale(0.5, 1.5),
        Tr.ResizeRandomCrop(cfg.DATA.CROP_SIZE), 
        Tr.RandomHFlip(0.5), 
        Tr.ColorJitter(0.5,0.5,0.5,0),
        Tr.Normalize_Caffe(),
    ])
    trainset = VOC_seg(cfg, tr_transforms)
    train_loader = DataLoader(trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    model = DeepLab_LargeFOV(cfg.DATA.NUM_CLASSES).cuda()
    
    params = model.get_params()
    lr = cfg.SOLVER.LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    optimizer = optim.SGD(
        params,
        lr=lr,
        weight_decay=wd,
        momentum=cfg.SOLVER.MOMENTUM
    ) # learning rate and weight decay is kept same for all the layers 
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    model.train()
    iterator = iter(train_loader)

    for it in range(1, cfg.SOLVER.MAX_ITER+1):
        try:
            sample = next(iterator)
        except:
            iterator = iter(train_loader)
            sample = next(iterator)
        img, masks = sample # VOC_seg dataloader returns image and the corresponing (pseudo) label
        logits = model(img.cuda(), img.size()) # passing image and image size to the forward method of model
        # logits is the CS or DP based feature maps of dimension (batch_size, num_classes, H, W), softmax is not applied to it
        
        mask_crf, mask_ret = masks
        # region S where both the masks are predicting same
        S = (mask_crf == mask_ret)

        H = nn.Softmax(dim=1)(logits) # applying softmax to the logits to obtain the probaility map for each class (21 classes)
        L_ce =
        
        logits = logits[...,0,0]
        fg_t = bboxes[:,-1][:,None].expand(bboxes.shape[0], np.prod(cfg.MODEL.ROI_SIZE))
        fg_t = fg_t.flatten().long()
        target = torch.zeros(logits.shape[0], dtype=torch.long)
        target[:fg_t.shape[0]] = fg_t
        loss = criterion(logits, target.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()



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