import cv2
import os.path
from utils.parser import AnnotationParser
import numpy as np
import albumentations as A
import albumentations.pytorch as AP
from utils.visualize  import *
from transforms import RandomExpand
from dataset import VOCDataset
from dataloader import od_collate_fn
import torch.utils.data as data
from trainer import Trainer
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from model import SSD
from losses import MultiBoxLoss
from trainer import Trainer

# configuration for data
root_dir = r'data\VOCdevkit\VOC2012'
anno_dir = r'data\VOCdevkit\VOC2012\Annotations'
img_dir = r'data\VOCdevkit\VOC2012\JPEGImages'

anno_parser = AnnotationParser()

# transform
train_transform = A.Compose([
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30),
    A.SmallestMaxSize(max_size=300),
    RandomExpand(0),
    A.SmallestMaxSize(max_size=300),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Resize(300,300),
    A.ToFloat(),
    AP.ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1, label_fields=['class_labels']))

val_transform = A.Compose([
    A.Resize(300,300),
    A.ToFloat(),
    AP.ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1, label_fields=['class_labels']))

#datasets
train_set = VOCDataset(dir=root_dir, anno_parser=anno_parser, train=True, transform=train_transform)
val_set = VOCDataset(dir=root_dir, anno_parser=anno_parser, train=False, transform=val_transform)

# dataloader
train_loader = data.DataLoader(dataset=train_set, batch_size=30, shuffle=False, collate_fn=od_collate_fn)
val_loader = data.DataLoader(dataset=val_set, batch_size=30, shuffle=True, collate_fn=od_collate_fn)
dataloader_dict = {'train': train_loader, 'val': val_loader}

# model
model = SSD()

# initialize parameters
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
model.extras.apply(weights_init)
model.location.apply(weights_init)
model.confidence.apply(weights_init)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# loss function
loss_fn = MultiBoxLoss()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

# trainer
trainer = Trainer(model, dataloader_dict=dataloader_dict, loss_fn=loss_fn, optimizer=optimizer, max_epochs=20)

trainer.train()

