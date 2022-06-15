from turtle import fillcolor
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize(img, bboxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    ax = plt.gca()
    for i, bbox in enumerate(bboxes):
        xy = (bbox[0], bbox[1])
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        ax.add_patch(plt.Rectangle(xy, w, h, fill=False, edgecolor='r', linewidth=2))


class Visualizer:
    def __init__(self, imaging, anno_parser):
        self.imaging = imaging
        self.anno_parser = anno_parser
    def show(self, img_path, anno_path):
        
        img = self.imaging.as_numpy(img_path, format='hwc', as_float=False)
        anno = self.anno_parser(anno_path)

        self.imshow(img, anno)

    def imshow(self, img, anno):
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        ax = plt.gca()
        bbox = anno[:, 1:5].astype(np.int32)
        
        for i, bb in enumerate(bbox):
            xy = (bb[0], bb[1])
            w = bb[2] - bb[0]
            h = bb[3] - bb[1]

            ax.add_patch(plt.Rectangle(xy, w, h, fill=False, edgecolor='r', linewidth=2))
    


