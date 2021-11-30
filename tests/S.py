import os
import torch
import numpy as np 
from PIL import Image

txt_name = 'train.txt'
f_path = os.path.join(cfg.DATA.ROOT, "ImageSets/Segmentation", txt_name)
img_path_crf  = os.path.join(cfg.DATA.ROOT, "Generation/Y_crf", "{}.png")
img_path_ret  = os.path.join(cfg.DATA.ROOT, "Generation/Y_ret", "{}.png")
filenames = [x.split('\n')[0] for x in open(f_path)]
for fn in filenames:
   img_crf = np.asarray(Image.open(img_path_crf.format(fn)))
   img_ret = np.asarray(Image.open(img_path_ret.format(fn)))
   S=np.where(img_crf==img_ret)
   
