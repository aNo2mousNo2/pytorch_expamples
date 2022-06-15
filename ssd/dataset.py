import os.path
import cv2
import numpy as np
import torch
import torch.utils.data as data
import cv2

class VOCDataset(data.Dataset):
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, dir, anno_parser, train=True, transform=None):
        self.root_dir = dir
        self.img_dir = os.path.join(self.root_dir, 'JPEGImages')
        self.anno_dir = os.path.join(self.root_dir, 'Annotations')
        self.imagesets_dir = os.path.join(self.root_dir, 'ImageSets')
        self.seg_dir = os.path.join(self.root_dir, 'Segmentation')

        self.anno_parser = anno_parser
        self.anno_parser.class_labels = self.classes
        self.train = train
        self.transform = transform

        self.ids, self.labels = self.get_set(type='main')

        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        return self.pull_item(index)
    
    def get_set(self, type='Main', class_name=''):
        class_name = class_name if class_name == '' else class_name + '_'
        fname = class_name + ('train' if self.train else 'val')

        fname = os.path.join(self.imagesets_dir, 'Main', fname + '.txt')

        ids = []
        labels = []
        for l in open(fname):
            items = l.split()
            id = items[0]
            label = np.array(items[1:])
            ids.append(id)
            labels.append(label)
        return ids, labels
            
    def pull_item(self, index):
        # image
        img_path = os.path.join(self.img_dir, self.ids[index] + '.jpg')
        img = cv2.cvtColor(cv2.imdecode(np.fromfile(img_path), flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w, c = img.shape

        # annotation
        anno_path = os.path.join(self.anno_dir, self.ids[index] + '.xml')
        names, bndboxes, difficults, poses = self.anno_parser(anno_path)

        # transform
        if self.transform is not None:
            transformed_data = self.transform(image=img, bboxes=bndboxes, class_labels=names)
            img, bndboxes, names = transformed_data['image'], transformed_data['bboxes'], transformed_data['class_labels']

        # img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
        # gt = np.hstack([boxes, np.expand_dims(labels, axis=1)])

        return img, bndboxes, names
    
