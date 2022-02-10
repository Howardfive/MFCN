import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils
from MFCN.tools import extract_image_patches,\
    reduce_mean, reduce_sum, same_padding



class DMFB(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DMFB, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.act_fn = nn.Sequential(nn.ReLU(), nn.BatchNorm2d(out_channel))
        self.conv_3 = nn.Conv2d(out_channel,in_channel, 3, 1, 1)
        conv_3_sets = []
        for _ in range(4):
            conv_3_sets.append(nn.Conv2d(in_channel,in_channel,3, padding=1))
        self.conv_3_sets = nn.ModuleList(conv_3_sets)
        self.conv_3_2 = nn.Conv2d(in_channel,in_channel, 3, padding=2, dilation=2)
        self.conv_3_4 = nn.Conv2d(in_channel,in_channel,3, padding=4, dilation=4)
        self.conv_3_8 = nn.Conv2d(in_channel,in_channel, 3, padding=8, dilation=8)
        self.norm = nn.BatchNorm2d(in_channel*4)
        self.conv_1 = nn.Conv2d(in_channel*4,out_channel, 1)

    def forward(self, inputs):
        src = inputs
        # conv-3
        x = self.act_fn(inputs)
        x = self.conv_3(x)
        K = []
        for i in range(4):
            if i != 0:
                p = eval('self.conv_3_' + str(2 ** i))(x)
                p = p + K[i - 1]
            else:
                p = x
            K.append(self.conv_3_sets[i](p))
        cat = torch.cat(K, 1)
        bottle = self.conv_1(self.norm(cat))
        out = bottle + src
        return out 


class ESA(nn.Module):
    def __init__(self,n_feats,treat = True):
        super(ESA,self).__init__()
        f = n_feats // 2
        self.treat = treat
        self.conv1 = nn.Conv2d(n_feats,f,kernel_size=1)
        self.conv_f = nn.Conv2d(f,f,kernel_size=1)
        self.conv_max = nn.Conv2d(f,f,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(f,f,kernel_size=3,stride=2,padding=0)
        self.conv3 = nn.Conv2d(f,f,kernel_size=3,padding=1)
        self.conv3_ = nn.Conv2d(f,f,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(f,n_feats,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.PReLU()

    def forward(self,x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1,kernel_size=5,stride=2)
        v_range = self.relu(self.conv_max(v_max))

        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3,(x.size(2),x.size(3)),mode = 'bilinear',align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        if self.treat:
            out = self.sigmoid(c4)
        else:
            out = c4
        return x*out
    

class DMSA(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DMSA,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DMFB = DMFB(self.in_channels,self.out_channels)
        self.ESA = ESA(self.out_channels)

    def forward(self,x):
        inputs = x
        x = self.DMFB(x)
        x = self.ESA(x)
        x += inputs
        return x


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,  padding=(kernel_size//2),stride=stride, bias=bias)



## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.PReLU(),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialAttention,self).__init__()
        assert kernel_size in (3,7),"kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        avgout = torch.mean(x,dim=1,keepdim=True)
        maxout,_=torch.max(x,dim=1,keepdim=True)
        x = torch.cat([avgout,maxout],dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBABlock(nn.Module):
    expansion = 1
    def __init__(self,inplane,plane,stride=1,downsample=None):
        super(CBABlock,self).__init__()
        self.conv1 = nn.Conv2d(inplane,plane,3,stride)
        self.bn1 = nn.BatchNorm2d(plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(plane,plane,3)
        self.bn2 = nn.BatchNorm2d(plane)
        self.ca = CALayer(plane)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class CBAM(nn.Module):
    def __init__(self,plane):
        self.ca = CALayer(plane)
        self.sa = SpatialAttention()

    def forward(self,x):
        x = self.ca(x) * x
        out = self.sa(x) * x
        return out

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self,  n_feat, kernel_size, conv=default_conv,  bias=True, bn=False, act=nn.PReLU()):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat,kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res