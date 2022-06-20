import numpy as np
import torch
import itertools

       

def get_dboxes():
    steps = [38, 19, 10, 5, 3, 1]
    ars1 = [1, 1/2, 2]
    ars2 = [1, 1/2, 2, 1/3, 3]
    aspect_ratios = [ars1, ars2, ars2, ars2, ars1, ars1]

    dboxes = []
    sizes = get_sizes()
    sizes_p = get_sizes_()

    for step, ars, s, s_p in zip(steps, aspect_ratios, sizes, sizes_p):
        centers = get_center_coords(step)
        s = ch_aspect_ratio(np.array([s]), aspect_ratio=ars)
        s = np.vstack([s, s_p]).reshape(-1, 2)
        for c, s in itertools.product(centers, s):
            dboxes.append(np.concatenate([c, s]))
    
    return torch.Tensor(dboxes).view(-1, 4)

def get_center_coords(step):
    centers = []
    d = 1 / step
    for i, j in itertools.product(range(step), repeat=2):
        centers.append([i * d + d / 2, j * d + d / 2])
    return centers


def get_sizes(s_min=0.2, s_max=0.9, m=6, k=6):
    """
    s[k] = s_min + (s_max - s_min) k / (m - 1)
    """
    s = s_min + (s_max - s_min) * np.arange(k) / (m - 1)
    return np.stack([s, s]).transpose(1, 0)


def get_sizes_(s_min=0.2, s_max=0.9, m=6, k=6):
    """
    s[k] = s_min + (s_max - s_min) * k / (m - 1)
    s_prime[k] = sqrt(s[k] * s[k+1])

    """
    s = get_sizes(s_min=s_min, s_max=s_max, m=m, k=k+1)
    s_prime = s[1:,:]
    s = s[:k,:]
    s_prime = np.sqrt(s * s_prime)
    return s_prime

def ch_aspect_ratio(source, aspect_ratio=[1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0]):
    return np.hstack([np.array([source[:, 0] / np.sqrt(ar), source[:, 1] * np.sqrt(ar)]) for ar in aspect_ratio]).transpose(1, 0)



        

        