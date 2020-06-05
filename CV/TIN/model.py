import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from torchsummary import summary

# from https://github.com/jannctu/TIN/blob/master/model.py
class Enrichment(nn.Module):
    def __init__(self, c_in, rate=2):
        super(Enrichment, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(c_in, 32, 3, padding=1)
        # 2 4 8 16?
        self.conv1 = nn.Conv2d(32, 32, 3, dilation=rate, padding=rate)
        self.conv2 = nn.Conv2d(32,32, 3, dilation=rate*2, padding=rate*2)
        self.conv3 = nn.Conv2d(32, 32, 3, dilation=rate*3, padding=rate*3)
        self.conv4 = nn.Conv2d(32, 32, 3, dilation=rate*4, padding=rate*4)
    
    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu(self.conv1(o))
        o2 = self.relu(self.conv2(o))
        o3 = self.relu(self.conv3(o))
        o4 = self.relu(self.conv4(o))
        out = o + o1 + o2 + o3 + o4 
        return out 

class TIN(nn.Module):
    def __init__(self, tin_m=2):
        super(TIN, self).__init__()
        self.tin_m = tin_m 
        # feature extractor, change to VGG or other model?
        self.conv1_1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv1_2 = nn.Conv2d(16,16, 3, padding=1)

        # Enrichment
        self.em1_1 = Enrichment(16, 4)
        self.em1_2 = Enrichment(16, 4)

        self.conv1_1_down = nn.Conv2d(32, 8, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(32, 8, 1, padding=0)

        # score
        self.score_stage1 = nn.Conv2d(8,1,1)

        self.relu = nn.ReLU(inplace=True)
        if tin_m > 1:
            self.conv2_1 = nn.Conv2d(16, 64, 3, padding=1)
            self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
            self.em2_1 = Enrichment(64, 4)
            self.em2_2 = Enrichment(64, 4)
            self.conv2_1_down  = nn.Conv2d(32, 8, 1, padding=0)
            self.conv2_2_down = nn.Conv2d(32, 8, 1, padding=0)
            self.score_stage2 = nn.Conv2d(8,1,1)


            self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
            self.score_final = nn.Conv2d(2,1,1)


    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        conv1_1_down = self.conv1_1_down(self.em1_1(conv1_1))
        conv1_2_down = self.conv1_2_down(self.em1_2(conv1_2))
        o1_out = self.score_stage1(conv1_1_down + conv1_2_down)
        if self.tin_m > 1:
            pool1 = self.maxpool(conv1_2)
            conv2_1 = self.relu(self.conv2_1(pool1))
            conv2_2 = self.relu(self.conv2_2(conv2_1))
            conv2_1_down = self.conv2_1_down(self.em2_1(conv2_1))
            conv2_2_down = self.conv2_2_down(self.em2_2(conv2_2))
            o2_out = self.score_stage2(conv2_1_down + conv2_2_down)
            upsample2 = nn.UpsamplingBilinear2d(size=(h,w))(o2_out)
            fuseout = torch.cat((o1_out, upsample2), dim=1)
            fuse = self.score_final(fuseout)
            results = [o1_out, upsample2, fuse]
            results = [torch.sigmoid(r) for r in results]
        else:
            results = [o1_out]
            results = [torch.sigmoid(r) for r in results]
        return results

model = TIN(tin_m=1)        
summary(model,(3,512,1024))

model = TIN(tin_m=2)        
summary(model,(3,512,1024))