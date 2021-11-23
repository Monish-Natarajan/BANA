import os
import collections
import numpy as np
from PIL import Image
from configs.defaults import _C

cfg = _C.clone()
cfg.merge_from_file('configs/stage2.yml')
cfg.freeze()

txt_name = 'train.txt'

f_path = os.path.join(cfg.DATA.ROOT, "ImageSets/Segmentation", txt_name)

img_path1a  = os.path.join(cfg.DATA.ROOT, "Generation/Y_crf", "{}.png")
img_path1b  = os.path.join(cfg.DATA.ROOT, "Generation/Y_ret", "{}.png")
img_path2  = os.path.join(cfg.DATA.ROOT, "SegmentationObject", "{}.png")

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

    img1a_np = asarray(img1a)
    img1b_np = asarray(img1b)
    img2_np = asarray(img2)

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

mean_iou_crf = mean_iou_crf/i
mean_iou_ret = mean_iou_ret/j
mean_iou_crf_ret = mean_iou_crf_ret/k

print(mean_iou_crf,"CRF")
print(mean_iou_ret,"RET")
print(mean_iou_crf_ret,"CRF_RET")