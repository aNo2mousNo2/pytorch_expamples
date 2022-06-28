from turtle import forward
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from defaultbox import DBox
from torch.autograd import Function

class SSD(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.num_classes = num_classes
        self.vgg = AtrousVGG()
        self.extras = Extras()
        self.l2norm = L2Norm()
        self.location = Location()
        self.confidence = Confidence(num_classes)
        self.dboxes = DBox()
    
    def forward(self, x):
        s1, s2 = self.vgg(x)
        s3, s4, s5, s6 = self.extras(s2)
        loc = self.location(s1, s2, s3, s4, s5, s6)
        conf = self.confidence(s1, s2, s3, s4, s5, s6)

        return loc, conf, self.dboxes.create(img_size=(300, 300))



class AtrousVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv15 = nn.Conv2d(1024, 1024, kernel_size=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x), inplace=True)
        x = F.relu(self.conv6(x), inplace=True)
        x = F.relu(self.conv7(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        x = F.relu(self.conv8(x), inplace=True)
        x = F.relu(self.conv9(x), inplace=True)
        y1 = F.relu(self.conv10(x), inplace=True)

        x = F.max_pool2d(y1, kernel_size=2, stride=2)
        x = F.relu(self.conv11(x), inplace=True)
        x = F.relu(self.conv12(x), inplace=True)
        x = F.relu(self.conv13(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = F.relu(self.conv14(x), inplace=True)
        y2 = F.relu(self.conv15(x), inplace=True)
        return y1, y2


class Extras(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv8 = nn.Conv2d(128, 256, kernel_size=3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        y1 = F.relu(self.conv2(x))
        x = F.relu(self.conv3(y1))
        y2 = F.relu(self.conv4(x))
        x = F.relu(self.conv5(y2))
        y3 = F.relu(self.conv6(x))
        x = F.relu(self.conv7(y3))
        y4 = F.relu(self.conv8(x))
        return y1, y2, y3, y4

class Location(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1024, 24, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 24, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 24, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 16, kernel_size=3, padding=1)
    
    def forward(self, x1, x2, x3, x4, x5, x6):
        y1 = self.conv1(x1).permute(0, 2, 3, 1).contiguous().view(x1.size(0), -1)
        y2 = self.conv2(x2).permute(0, 2, 3, 1).contiguous().view(x2.size(0), -1)
        y3 = self.conv3(x3).permute(0, 2, 3, 1).contiguous().view(x3.size(0), -1)
        y4 = self.conv4(x4).permute(0, 2, 3, 1).contiguous().view(x4.size(0), -1)
        y5 = self.conv5(x5).permute(0, 2, 3, 1).contiguous().view(x5.size(0), -1)
        y6 = self.conv6(x6).permute(0, 2, 3, 1).contiguous().view(x6.size(0), -1)
        y = torch.cat([y1, y2, y3, y4, y5, y6], 1)
        return y.view(y.size(0), -1, 4)


class Confidence(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)
    
    def forward(self, x1, x2, x3, x4, x5, x6):
        y1 = self.conv1(x1).permute(0, 2, 3, 1).contiguous().view(x1.size(0), -1)
        y2 = self.conv2(x2).permute(0, 2, 3, 1).contiguous().view(x2.size(0), -1)
        y3 = self.conv3(x3).permute(0, 2, 3, 1).contiguous().view(x3.size(0), -1)
        y4 = self.conv4(x4).permute(0, 2, 3, 1).contiguous().view(x4.size(0), -1)
        y5 = self.conv5(x5).permute(0, 2, 3, 1).contiguous().view(x5.size(0), -1)
        y6 = self.conv6(x6).permute(0, 2, 3, 1).contiguous().view(x6.size(0), -1)
        y = torch.cat([y1, y2, y3, y4, y5, y6], 1)
        return y.view(y.size(0), -1, self.num_classes)

class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_params()
        self.eps = 1e-10
    
    def reset_params(self):
        init.constant_(self.weight, self.scale)
    
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        x = torch.div(x, norm + self.eps)

        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        return weights * x

class Detect(Function):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh