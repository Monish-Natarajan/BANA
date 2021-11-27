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

img_path1a  = os.path.join(cfg.DATA.ROOT, "Generation/Y_crf", "{}.png")
img_path1b  = os.path.join(cfg.DATA.ROOT, "Generation/Y_ret", "{}.png")
img_path2  = os.path.join(cfg.DATA.ROOT, "SegmentationClass", "{}.png")
jpeg_path = os.path.join(cfg.DATA.ROOT, "JPEGImages", "{}.jpg")

filenames = [x.split('\n')[0] for x in open(f_path)]

mean_iou_crf = 0
mean_iou_ret = 0
mean_iou_crf_ret = 0
i=0
j=0
k=0
for fn in filenames:
    img1a = Image.open(img_path1a.format(fn))
    img1b = Image.open(img_path1b.format(fn))
    img2 = Image.open(img_path2.format(fn))
    jpeg = Image.open(jpeg_path.format(fn))

    img1a_np = np.asarray(img1a)
    img1b_np = np.asarray(img1b)
    img2_np = np.asarray(img2)
    jpeg_np = np.asarray(jpeg)

    intersection_a = np.logical_and(img1a_np,img2_np)
    union_a = np.logical_or(img1a_np,img2_np)
    union_sum_a = np.sum(union_a)

    intersection_b = np.logical_and(img1b_np,img2_np)
    union_b = np.logical_or(img1b_np,img2_np)
    union_sum_b = np.sum(union_b)

    img1_np = np.logical_or(img1a_np, img1b_np)
    intersection = np.logical_and(img1_np,img2_np)
    union = np.logical_or(img1_np,img2_np)
    union_sum = np.sum(union)
  
    if union_sum != 0:
        mean_iou_crf_ret = np.sum(intersection)/union_sum + mean_iou_crf_ret
        k+=1
        #print(mean_iou_crf_ret/k)

    if union_sum_a != 0:
        mean_iou_crf = np.sum(intersection_a)/union_sum_a + mean_iou_crf
        i+=1
        #print(mean_iou_crf/i)

    if union_sum_b != 0:
        mean_iou_ret = np.sum(intersection_b)/union_sum_b + mean_iou_ret
        j+=1
        #print(mean_iou_ret/j)


    mask_img1a = wandb.Image(jpeg, masks={
    "predictions": {
        "mask_data": img1a_np,
    },
    "ground_truth": {
        "mask_data": img2_np,
    },})

    mask_img1b = wandb.Image(jpeg, masks={
    "predictions": {
        "mask_data": img1b_np,
    },
    "ground_truth": {
        "mask_data": img2_np,
    },})

    mask_img1 = wandb.Image(jpeg, masks={
    "predictions": {
        "mask_data": img1_np,
    },
    "ground_truth": {
        "mask_data": img2_np,
    },})

    if i < 25: 
        wandb.log({"CRF": mask_img1a})
        wandb.log({"RET": mask_img1b})
        wandb.log({"CRF-RET": mask_img1})

mean_iou_crf = mean_iou_crf/i
mean_iou_ret = mean_iou_ret/j
mean_iou_crf_ret = mean_iou_crf_ret/k

print(i, j, k)

print(mean_iou_crf,"CRF")
print(mean_iou_ret,"RET")
print(mean_iou_crf_ret,"CRF_RET")
wandb.finish()