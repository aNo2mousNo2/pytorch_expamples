from matplotlib.transforms import Transform
import numpy as np
import torchvision.transforms as transforms
from albumentations.augmentations.bbox_utils import denormalize_bbox
from albumentations.core.transforms_interface import DualTransform, to_tuple
import albumentations as A
from typing import Sequence
from albumentations.augmentations.functional import pad_with_params
import random


class RandomExpand(DualTransform):
    def __init__(
        self,
        mean: float,
        expand_limit=1.5,
        always_apply: bool=False,
        p: float=1.0
    ):
        super(RandomExpand, self).__init__(always_apply, p)
        self.mean = mean
        self.expand_limit = expand_limit
    
    def apply(self, img, ratio=1.5, left=0, top=0, **params):
        h, w, c = img.shape
        left *= w * ratio
        top *= h * ratio

        expand_img = np.zeros((int(h * ratio), int(w * ratio), c), dtype=np.uint8)
        expand_img[:,:,:] = self.mean
        expand_img[int(top):int(top + h), int(left):int(left + w), :] = img
        img = expand_img
        
        return img
        
    def apply_to_bbox(self, bbox: Sequence[float], ratio=1.5, left=0, top=0, **params):
        xmin, ymin, xmax, ymax = bbox[:4]
        w = xmax - xmin
        h = ymax - ymin
        xmin = xmin / ratio + left
        ymin = ymin / ratio + top
        xmax = xmin + w / ratio
        ymax = ymin + h / ratio
        
        return (xmin, ymin, xmax, ymax) + bbox[4:]
            
    def get_params(self):
        ratio = random.uniform(1.0, self.expand_limit)
        left = random.uniform(0, 1 - 1 / ratio)
        top = random.uniform(0, 1 - 1 / ratio)
        return { 'ratio': ratio, 'left': left, 'top': top }


class AlRandomExpand:
    def __init__(self, mean):
        self.mean = mean
    
    def __call__(self, image, anno):
        if np.random.randint(2):
            return image, anno
        
        c, h, w = image.shape
        ratio = np.random.uniform(1, 4)
        left = np.random.uniform(0, w * ratio - w)
        top = np.random.uniform(0, h * ratio - h)

        expand_img = np.zeros((c, int(h * ratio), int(w * ratio)), dtype=np.int32)
        expand_img[:,:,:] = self.mean
        expand_img[:,int(top):int(top + h), int(left):int(left + w)] = image
        image = expand_img
        
        anno = anno.copy()
        anno[:, 1:5] += np.array([[int(left), int(top), int(left), int(top)]])
        
        return image, anno