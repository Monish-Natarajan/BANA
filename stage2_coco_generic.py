import os
import sys
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

import data.transforms_bbox as Tr
from data.coco import COCO_box
from configs.defaults import _C
from models.ClsNet import Labeler, pad_for_grid
from utils.densecrf import DENSE_CRF

logger = logging.getLogger("stage2")

import wandb


def main(cfg):
    if cfg.SEED:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)

    tr_transforms = Tr.Normalize_Caffe()

    #COCO Specific Code
    ann_path =  os.path.join(cfg.DATA.ROOT,'annotations/instances_val2017.json')
    data_root = os.path.join(cfg.DATA.ROOT,'val2017')
    
    trainset = COCO_box(data_root,ann_path, tr_transforms)
    train_loader = DataLoader(trainset, batch_size=1)
    
    model = Labeler(cfg.DATA.NUM_CLASSES, cfg.MODEL.ROI_SIZE, cfg.MODEL.GRID_SIZE).cuda()

    # Restore the model saved on WandB
    model_stage_1 = wandb.restore('weights/ClsNet.pt', run_path='dl-segmentation/MLRC-BANA/3tlmc1pv')
    model.load_state_dict(torch.load(model_stage_1.name))

    WEIGHTS = torch.clone(model.classifier.weight.data)
    model.eval()
    
    bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std = cfg.MODEL.DCRF
    dCRF = DENSE_CRF(bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
    
    if cfg.SAVE_PSEUDO_LABLES:
        folder_name = os.path.join('/kaggle/working', cfg.NAME)
        os.mkdir(folder_name)
        save_paths = []
        for txt in ("Y_crf_COCO", "Y_ret_COCO"):
            sub_folder = folder_name + f"/{txt}"
            os.mkdir(sub_folder)
            save_paths += [os.path.join(sub_folder, "{}.png")]
            
    logger.info(f"START {cfg.NAME} -->")
    with torch.no_grad():
        for it, (img, bboxes, bg_mask) in enumerate(tqdm(train_loader)):
            '''
            img     : (1,3,H,W) float32
            bboxes  : (1,K,5)   float32
            bg_mask : (1,H,W)   float32
            '''
            fn,rgb_img_path = trainset.filename(it)
            rgb_img = np.array(Image.open(rgb_img_path))

            bboxes = bboxes[0] # (1,K,5) --> (K,5)
            bg_mask = bg_mask[None] # (1,H,W) --> (1,1,H,W)
            img_H, img_W = img.shape[-2:]
            norm_H, norm_W = (img_H-1)/2, (img_W-1)/2
            bboxes[:,[0,2]] = bboxes[:,[0,2]]*norm_W + norm_W
            bboxes[:,[1,3]] = bboxes[:,[1,3]]*norm_H + norm_H
            bboxes = bboxes.long()
            gt_labels = bboxes[:,4].unique()
            
            features = model.get_features(img.cuda())
            features = F.interpolate(features, img.shape[-2:], mode='bilinear', align_corners=True)
            padded_features = pad_for_grid(features, cfg.MODEL.GRID_SIZE)
            padded_bg_mask = pad_for_grid(bg_mask.cuda(), cfg.MODEL.GRID_SIZE)
            grid_bg, valid_gridIDs = model.get_grid_bg_and_IDs(padded_bg_mask, cfg.MODEL.GRID_SIZE)
            bg_protos = model.get_bg_prototypes(padded_features, padded_bg_mask, grid_bg, cfg.MODEL.GRID_SIZE)
            bg_protos = bg_protos[0,valid_gridIDs] # (1,GS**2,dims,1,1) --> (len(valid_gridIDs),dims,1,1)
            normed_bg_p = F.normalize(bg_protos)
            normed_f = F.normalize(features)
            bg_attns = F.relu(torch.sum(normed_bg_p*normed_f, dim=1))
            bg_attn = torch.mean(bg_attns, dim=0, keepdim=True) # (len(valid_gridIDs),H,W) --> (1,H,W)
            bg_attn[bg_attn < cfg.MODEL.BG_THRESHOLD * bg_attn.max()] = 0
            Bg_unary = torch.clone(bg_mask[0]) # (1,H,W)
            region_inside_bboxes = Bg_unary[0]==0 # (H,W)
            Bg_unary[:,region_inside_bboxes] = bg_attn[:,region_inside_bboxes].detach().cpu()
            
            # Fg_unary = []
            # for uni_cls in gt_labels:
            #     w_c = WEIGHTS[uni_cls][None]
            #     raw_cam = F.relu(torch.sum(w_c*features, dim=1)) # (1,H,W)
            #     normed_cam = torch.zeros_like(raw_cam)
            #     for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
            #         denom = raw_cam[:,hmin:hmax,wmin:wmax].max() + 1e-12
            #         normed_cam[:,hmin:hmax,wmin:wmax] = raw_cam[:,hmin:hmax,wmin:wmax] / denom
            #     Fg_unary += [normed_cam]

            N = len(gt_labels)
            Fg_unary = 1 - Bg_unary
            Fg_unary = torch.cat([Fg_unary]*N, dim=0).detach().cpu()

            unary = torch.cat((Bg_unary,Fg_unary), dim=0)
            unary[:,region_inside_bboxes] = torch.softmax(unary[:,region_inside_bboxes], dim=0)
            refined_unary = dCRF.inference(rgb_img, unary.numpy())

            #print("Fg_unary Fg_unary refined_unary",Fg_unary.shape,Bg_unary.shape,refined_unary.shape)
            
            # (Out of bboxes) reset Fg scores to zero
            for idx_cls, uni_cls in enumerate(gt_labels,1):
                mask = np.zeros((img_H,img_W))
                for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                    mask[hmin:hmax,wmin:wmax] = 1
                refined_unary[idx_cls] *= mask

            # Y_crf
            tmp_mask = refined_unary.argmax(0)
            Y_crf = np.zeros_like(tmp_mask, dtype=np.uint8)
            for idx_cls, uni_cls in enumerate(gt_labels,1):
                Y_crf[tmp_mask==idx_cls] = uni_cls
            Y_crf[tmp_mask==0] = 0
            
            # Y_ret
            tmp_Y_crf = torch.from_numpy(Y_crf) # (H,W)
            gt_labels_with_Bg = [0] + gt_labels.tolist()
            corr_maps = []
            for uni_cls in gt_labels_with_Bg:
                indices = tmp_Y_crf==uni_cls

                # if indices.sum():
                #     normed_p = F.normalize(features[...,indices].mean(dim=-1))   # (1,dims)
                #     corr = F.relu((normed_f*normed_p[...,None,None]).sum(dim=1)) # (1,H,W)
                # else:
                #     normed_w = F.normalize(WEIGHTS[uni_cls][None])
                #     corr = F.relu((normed_f*normed_w).sum(dim=1)) # (1,H,W)

                normed_p = F.normalize(features[...,indices].mean(dim=-1))   # (1,dims)
                corr = F.relu((normed_f*normed_p[...,None,None]).sum(dim=1)) # (1,H,W)    
                corr_maps.append(corr)
            corr_maps = torch.cat(corr_maps) # (1+len(gt_labels),H,W)
            
            # (Out of bboxes) reset Fg correlations to zero
            for idx_cls, uni_cls in enumerate(gt_labels_with_Bg):
                if uni_cls == 0:
                    corr_maps[idx_cls, ~region_inside_bboxes] = 1
                else:
                    mask = torch.zeros(img_H,img_W).type_as(corr_maps)
                    for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                        mask[hmin:hmax,wmin:wmax] = 1
                    corr_maps[idx_cls] *= mask

            tmp_mask = corr_maps.argmax(0).detach().cpu().numpy()
            Y_ret = np.zeros_like(tmp_mask, dtype=np.uint8)
            for idx_cls, uni_cls in enumerate(gt_labels,1):
                Y_ret[tmp_mask==idx_cls] = uni_cls
            Y_ret[tmp_mask==0] = 0
            
            if cfg.SAVE_PSEUDO_LABLES:
                for pseudo, save_path in zip([Y_crf, Y_ret], save_paths):
                    Image.fromarray(pseudo).save(save_path.format(fn))

    logger.info(f"END {cfg.NAME} -->")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    main(cfg)