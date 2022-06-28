import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from albumentations.core.utils 

class MultiBoxLoss(nn.Module):
    def __init__(self, max_iou=0.5, neg_pos=3, device='cpu'):
        super().__init__()
        self.max_iou = max_iou
        self.negpos_ratio = neg_pos
        self.device = device
        
    
    def forward(self, predictions, target_bboxes, target_labels):
        
        loc_data, conf_data, dboxes = predictions
        batch_size = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)

        # target labels
        conf_t_label = torch.LongTensor(batch_size, num_dbox).to(self.device)
        # target offsets
        locs = torch.LongTensor(batch_size, num_dbox, 4).to(self.device)

        for idx in range(batch_size):
            tb = target_bboxes[idx].to(self.device)
            tl = target_labels[idx].to(self.device)
            dboxes = dboxes.to(self.device)

            iou = get_iou(tb, dboxes)
            best_iou_for_dbox, best_idx_for_dbox = iou.max(0)
            best_iou_for_bbox, best_idx_for_bbox = iou.max(1)

            # TODO: if the best dbox for a bbox is not match the best bbox for the dbox 

            best_bbox_for_dbox = tb[best_idx_for_dbox]

            c = tl[best_idx_for_dbox] + 1 # shift labels to define background label as 0
            c[best_iou_for_dbox < 0.5] = 0 # set best labels to 0(background) for dbox the iou is less than the threshold
            
            # calculate offset
            l = get_offset(best_bbox_for_dbox, dboxes)

            conf_t_label[idx] = c
            locs[idx] = l
        

        positive_dbox_mask = conf_t_label > 0
        positive_offset_idx = positive_dbox_mask.unsqueeze(positive_dbox_mask.dim()).expand_as(loc_data)

        # offsets for positive dboxes
        p_loc = loc_data[positive_offset_idx].view(-1, 4) # predictions
        t_loc = locs[positive_offset_idx].view(-1, 4) # targets

        # loss for offsets
        loss_l = F.smooth_l1_loss(p_loc, t_loc, reduction='sum')

        # prediction labels
        batch_conf = conf_data.view(-1, num_classes)

        # loss for labels
        loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction='none')

        num_pos = positive_dbox_mask.long().sum(1, keepdim=True)
        loss_c = loss_c.view(batch_size, -1)
        loss_c[positive_dbox_mask] = 0
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_neg = torch.clamp(num_pos * self.negpos_ratio, max=num_dbox)
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        pos_idx_mask = positive_dbox_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        conf_hnm = conf_data[(pos_idx_mask + neg_idx_mask).gt(0)].view(-1, num_classes)

        conf_t_label_hnm = conf_t_label[(positive_dbox_mask + neg_mask).gt(0)]
        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

        N = num_pos.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
        

def nearest(t_bboxes, dboxes):
    iou = iou(t_bboxes, dboxes)
    top_dbox_iou, top_dbox_idx = iou.max(1)
    top_bbox_iou, top_bbox_idx = iou.max(0)
    conf = t_bboxes[top_bbox_idx]
    return conf

def max_iou(b1, b2, axis=0):
    """
    Return:
        (top_iou, top_idx):
        max IoUs and the indeces in the boxes specified by axis for each another box
    """
    iou = get_iou(b1, b2)
    top_iou, top_idx = iou.max(axis)
    return top_iou, top_idx

def get_offset(t_bboxes, dboxes):
    d_size = dboxes[:, 2:] - dboxes[:, :2]
    t_size = t_bboxes[:, 2:] - t_bboxes[:, :2]
    diff = t_bboxes - dboxes
    dcx = (diff[:, 0] + diff[:, 2]) * 0.5 * 10 / d_size[:, 0]
    dxy = (diff[:, 1] + diff[:, 3]) * 0.5 * 10 / d_size[:, 1]
    r = t_size / d_size
    dwh = torch.log(r) * 5
    return torch.cat([dcx.unsqueeze(1), dxy.unsqueeze(1), dwh], 1)

def get_iou(b1, b2):
    """
    b1, b2: (x_min, y_min, x_max, y_max) normilized
    """
    i = intersect(b1, b2)
    s1 = area(b1).unsqueeze(1).expand_as(i)
    s2 = area(b2).unsqueeze(0).expand_as(i)
    return i / (s1 + s2 - i)

def area(b):
    wh = b[:, :2] - b[:, 2:]
    return wh[:, 0] * wh[:, 1]

def intersect(b1, b2):
    try:
        num_b1 = b1.size(0)
        num_b2 = b2.size(0)
        min = torch.max(b1[:, :2].unsqueeze(1).expand(num_b1, num_b2, 2), b2[:, :2].unsqueeze(0).expand(num_b1, num_b2, 2))
        max = torch.min(b1[:, 2:].unsqueeze(1).expand(num_b1, num_b2, 2), b2[:, 2:].unsqueeze(0).expand(num_b1, num_b2, 2))
        wh = torch.clamp((max - min), min=0)
        return wh[:, :, 0] * wh[:, :, 1]
    except Exception as e:
        print(e)


