import os
import pandas as pd
import numpy as np
from PIL import Image
import xml.etree.cElementTree as et

root_dir = os.getcwd() + '/VOCdevkit/VOC2012/'
directory = os.path.join(root_dir,"Annotations")

for filename in os.listdir(directory):
    file_annot = os.path.join(directory,filename)
    tree=et.parse(file_annot)
    root=tree.getroot()
    bounding_boxes = []
    img_size = []
    for size in root.iter('size'):
        for width in size.iter('width'):
            w = int(width.text)
            img_size.append(w)
        for height in root.iter('height'):
            h = int(height.text)
            img_size.append(h)
    for bndbox in root.iter('bndbox'):
        bbox = [0]*4
        
        for xmin in bndbox.iter('xmin'):
            x_min = int(float(xmin.text))
        for xmax in bndbox.iter('xmax'):
            x_max = int(float(xmax.text))
        for ymin in bndbox.iter('ymin'):
            y_min = int(float(ymin.text))
        for ymax in bndbox.iter('ymax'):
            y_max = int(float(ymax.text))
        bbox[0] = x_min
        bbox[1] = y_min
        bbox[2] = x_max
        bbox[3] = y_max
        bbox = tuple(bbox)
        bounding_boxes.append(bbox)
    #flipped dimensions here
    mask = np.ones((img_size[1],img_size[0]),dtype=np.uint8)
    for bbox in bounding_boxes:
        (xmin,ymin,xmax,ymax) = bbox
        mask[ymin:ymax,xmin:xmax] = 0
    
    im = Image.fromarray(mask)
    output_name = filename[:-4] + ".png"
    output_dir = os.path.join(root_dir,"BgMaskfromBoxes")
    output_path = os.path.join(output_dir,output_name)
    print(output_name)
    im.save(output_path)