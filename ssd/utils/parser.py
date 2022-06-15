import xml.etree.ElementTree
import numpy as np

class AnnotationParser:
    def __init__(self, class_labels=None, enable_difficult=True, enable_bndbox=True, enable_pose=True):
        self.enable_difficult = enable_difficult
        self.enable_bndbox = enable_bndbox
        self.enable_pose = enable_pose
        self.class_labels = class_labels
    
    def __call__(self, path):
        return self.parse_annotation(path)

    def parse_annotation(self, path):
        x = xml.etree.ElementTree.parse(path)

        labels = []
        bndboxes = []
        poses = []
        difficults = []
        for obj in x.iter('object'):
            name = obj.find('name').text.lower().strip()
            name = self.class_labels.index(name) if self.class_labels is not None else name
            labels.append(name)
            difficults.append(bool(obj.find('difficult').text) if self.enable_difficult else None)
            poses.append(obj.find('pose').text.lower().strip() if self.enable_pose else None)
            if self.enable_bndbox:
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                bndboxes.append(np.array([xmin, ymin, xmax, ymax]))
    
        return labels, bndboxes, difficults, poses