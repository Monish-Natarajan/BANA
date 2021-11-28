import os
import collections
import numpy as np
from PIL import Image
from configs.defaults import _C
import wandb

wandb.init()

cfg = _C.clone()
cfg.merge_from_file('configs/stage2.yml')
cfg.freeze()

txt_name = 'train.txt'

f_path = os.path.join(cfg.DATA.ROOT, "ImageSets/Segmentation", txt_name)

img_path_crf  = os.path.join(cfg.DATA.ROOT, "Generation/Y_crf", "{}.png")
img_path_ret  = os.path.join(cfg.DATA.ROOT, "Generation/Y_ret", "{}.png")
img_path_crf_u0  = os.path.join(cfg.DATA.ROOT, "Generation/Y_crf_u0", "{}.png")

img_path_gt  = os.path.join(cfg.DATA.ROOT, "SegmentationClass", "{}.png")
jpeg_path = os.path.join(cfg.DATA.ROOT, "JPEGImages", "{}.jpg")

filenames = [x.split('\n')[0] for x in open(f_path)]

mean_iou_crf = 0
mean_iou_ret = 0
mean_iou_crf_u0 = 0
mean_iou_crf_ret = 0

i=0

for fn in filenames:
    img_crf = Image.open(img_path_crf.format(fn))
    img_ret = Image.open(img_path_ret.format(fn))
    img_crf_u0 = Image.open(img_path_crf_u0.format(fn))
    img_gt = Image.open(img_path_gt.format(fn))
    jpeg = Image.open(jpeg_path.format(fn))

    img_crf_np = np.asarray(img_crf)
    img_crf_u0_np = np.asarray(img_crf_u0)
    img_ret_np = np.asarray(img_ret)
    img_gt_np = np.asarray(img_gt)
    jpeg_np = np.asarray(jpeg)

    intersection_crf = np.logical_and(img_crf_np, img_gt_np)
    union_crf = np.logical_or(img_crf_np, img_gt_np)
    union_sum_crf = np.sum(union_crf)

    intersection_crf_u0 = np.logical_and(img_crf_u0_np, img_gt_np)
    union_crf_u0 = np.logical_or(img_crf_u0_np, img_gt_np)
    union_sum_crf_u0 = np.sum(union_crf_u0)

    intersection_ret = np.logical_and(img_ret_np, img_gt_np)
    union_ret = np.logical_or(img_ret_np, img_gt_np)
    union_sum_ret = np.sum(union_ret)

    img_crf_ret = np.logical_or(img_crf_np, img_ret_np)
    intersection_crf_ret = np.logical_and(img_crf_ret, img_gt_np)
    union_crf_ret = np.logical_or(img_crf_ret, img_gt_np)
    union_sum_crf_ret = np.sum(union_crf_ret)
  
    if union_sum_crf_ret != 0:
        mean_iou_crf_ret = np.sum(intersection_crf_ret)/union_sum_crf_ret + mean_iou_crf_ret

    if union_sum_crf != 0:
        mean_iou_crf = np.sum(intersection_crf)/union_sum_crf + mean_iou_crf

    if union_sum_crf_u0 != 0:
        mean_iou_crf_u0 = np.sum(intersection_crf_u0)/union_sum_crf_u0 + mean_iou_crf_u0

    if union_sum_ret != 0:
        mean_iou_ret = np.sum(intersection_ret)/union_sum_ret + mean_iou_ret


    mask_img_crf = wandb.Image(jpeg, masks={
    "predictions": {
        "mask_data": img_crf_np,
    },
    "ground_truth": {
        "mask_data": img_gt_np,
    },})

    mask_img_crf_u0 = wandb.Image(jpeg, masks={
    "predictions": {
        "mask_data": img_crf_u0_np,
    },
    "ground_truth": {
        "mask_data": img_gt_np,
    },})

    mask_img_ret = wandb.Image(jpeg, masks={
    "predictions": {
        "mask_data": img_ret_np,
    },
    "ground_truth": {
        "mask_data": img_gt_np,
    },})

    mask_img_crf_ret = wandb.Image(jpeg, masks={
    "predictions": {
        "mask_data": img_crf_ret,
    },
    "ground_truth": {
        "mask_data": img_gt_np,
    },})

    if i < 25: 
        wandb.log({"CRF": mask_img_crf})
        wandb.log({"CRF_u0": mask_img_crf_u0})
        wandb.log({"RET": mask_img_ret})
        wandb.log({"CRF-RET": mask_img_crf_ret})
    i += 1

mean_iou_crf = mean_iou_crf/i
mean_iou_crf_u0 = mean_iou_crf_u0/i
mean_iou_ret = mean_iou_ret/i
mean_iou_crf_ret = mean_iou_crf_ret/i


print(mean_iou_crf,"CRF")
print(mean_iou_crf_u0,"CRF_u0")
print(mean_iou_ret,"RET")
print(mean_iou_crf_ret,"CRF_RET")
wandb.finish()