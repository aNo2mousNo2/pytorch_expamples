import torch

def od_collate_fn(batch):
    boxes = []
    labels = []
    imgs = []
    for sample in batch:
        imgs.append(torch.FloatTensor(sample[0]))
        boxes.append(torch.FloatTensor(sample[1]))
        labels.append(torch.FloatTensor(sample[2]))
    
    imgs = torch.stack(imgs, dim=0)

    return imgs, boxes, labels