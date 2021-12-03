import matplotlib.pyplot as plt
import numpy as np
import os
from pycocotools.coco import COCO
import skimage.io as io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm
from pathlib import Path
import random
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple

class COCO_Segmentation(Dataset):
    def __init__(
        self, 
        annotations: COCO, 
        img_ids: List[int], 
        cat_ids: List[int], 
        root_path: Path, 
        transform: Optional[Callable]=None
    ) -> None:
        super().__init__()
        self.annotations = annotations
        self.img_data = annotations.loadImgs(img_ids)
        self.cat_ids = cat_ids
        self.files = [os.path.join(root_path,img["file_name"]) for img in self.img_data]
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        ann_ids = self.annotations.getAnnIds(
            imgIds=self.img_data[i]['id'], 
            catIds=self.cat_ids, 
            iscrowd=None
        )
        anns = self.annotations.loadAnns(ann_ids)
        # mask = torch.LongTensor(np.max(np.stack([self.annotations.annToMask(ann) * ann["category_id"] 
        #                                          for ann in anns]), axis=0)).unsqueeze(0)
        mask = np.max(np.stack([self.annotations.annToMask(ann) * ann["category_id"] 
                                                  for ann in anns]), axis=0)

        img = np.asarray(Image.open(self.files[i]))
        # if img.shape[0] == 1:
        #     img = torch.cat([img]*3)
        
        if self.transform is not None:
            return self.transform(img, mask)
        
        return img, mask