import functools
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
torch.nn.Module.dump_patches = True
torch.backends.cudnn.benchmark = True
import numpy as np
import PIL
from PIL import Image
import imageio
from torchvision import transforms
from MFCN.SA_Models import RCAB
from torch.autograd import Variable



class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)
    

class FFM(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(FFM,self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g,F_int,kernel_size = 1,stride=1 ,padding=0), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l,F_int,kernel_size=1,stride=1,padding=0), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int,1,kernel_size=1,stride=1,padding=0), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.PReLU()

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi


def make_layer(block,n_layers):
    layers=[]
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)



class BranchNetwork(nn.Module):
    def __init__(self, opt):
        super(BranchNetwork, self).__init__()
        self.opt=opt
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)  # 1 layers
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):   # 5-2=3 layers
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)

        self.atten = RCAB(max(N,opt.min_nfc),1)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),3,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh())

    def forward(self,x):
        x1 = self.head(x)
        x2 = self.body(x1)
        x3 =self.atten(x2)
        x4 = self.tail(x3)
        return x4


class G(nn.Module):
    def __init__(self,opt,m=0.999):
        super(G,self).__init__()
        self.opt = opt
        N = int(opt.nfc)
        self.FFM = FFM(F_g=max(2*N,opt.min_nfc),F_l=max(N,opt.min_nfc),F_int=max(N,opt.min_nfc))

        self.Main = BranchNetwork(opt)
        self.Momentum = BranchNetwork(opt)
        self.m = m
        self.transform_horizontal = transforms.RandomHorizontalFlip(p=1)
        self.transform_Vertical = transforms.RandomVerticalFlip(p=1)
        self.tvloss = TVLoss()

        for para_1,para_2 in zip(self.Main.parameters(),self.Momentum.parameters()):
            para_2.data.copy_(para_1.data)
            para_2.requires_grad = False

    @torch.no_grad()
    def _momentum_update_Momentum(self):
        for para_1,para_2 in zip(self.Main.parameters(),self.Momentum.parameters()):
            para_2.data = para_2.data * self.m + para_1.data *(1.-self.m)
    
    def tensor_to_PIL(self,x):
        unloader = transforms.ToPILImage()
        img = x.cpu().clone()
        img = img.squeeze(0)
        img = unloader(img)
        return img

    def PIL_to_Image(self,x):
        loader = transforms.Compose([transforms.ToTensor()])
        img = loader(x).unsqueeze(0)
        return img.to(torch.device('cuda'))

    def forward(self,x,y):
        inputs = x
        x1 = self.Main(x)
        ind = int((y.shape[2]-x1.shape[2])/2)  
        y1 = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        part1 = x1 +y1
        with torch.no_grad():
            self._momentum_update_Momentum()
            img = self.tensor_to_PIL(inputs)  
            img = self.transform_horizontal(img)
            img = self.PIL_to_Image(img)
            x2 = self.Momentum(img)
        ind1 = int((y.shape[2]-x2.shape[2])/2)   
        y2 = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        part2 = x2 +y2
        criterion = nn.SmoothL1Loss()
        loss = criterion(part1,part2.detach())
        out = Variable(part1,requires_grad=True)
        out = self.tvloss(out)
        out.backward(torch.cuda.FloatTensor([1,0.1,0.01]).mean(),retain_graph=True)
        return part1,loss



class ResBlock(nn.Module):
    """----conv ------ReLu ----- conv ------
              |_______________________|
    """
    def __init__(self,in_channel,out_channel,padd,stride, res_scale=1,is_bn=False):
        super(ResBlock, self).__init__()
        self.is_bn = is_bn
        self.conv = nn.Conv2d(in_channel ,in_channel,kernel_size=3,stride=stride,padding=1)
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channel ,in_channel,kernel_size=3,stride=stride,padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.conv(x)
        if not self.is_bn:
            res = self.bn(res)
            res = self.conv3(self.relu(res)).mul(self.res_scale)
        else:
            res = self.conv3(self.relu(res)).mul(self.res_scale)
        res += x
        return res


class BasicBlock(nn.Sequential):
    """conv + PReLu"""
    def __init__(self,  in_channels, out_channels, padd, stride=1, bias=True, bn=False, act=nn.PReLU()):
        m = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padd,stride=stride, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)



class Block(nn.Sequential):
    def __init__(self,in_channel,out_channel,stride):
        super(Block,self).__init__()
        self.add_module("DepthConv",nn.Conv2d(in_channel,in_channel,kernel_size = 3,stride=1,padding=0,groups=in_channel)),
        self.add_module("pointConv",nn.Conv2d(in_channel,out_channel,kernel_size = 1,stride=1,padding=0,groups=1)),
        self.add_module("Norm",nn.BatchNorm2d(out_channel)),
        self.add_module("LeaykRelu",nn.LeakyReLU(0.2,inplace=True))

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),    
        self.add_module('norm',nn.BatchNorm2d(out_channel)),    
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))    


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   

def initialize_weights(self):
    for m in self.modules:
        if isinstance(m,nn.Conv2d):
            m.init.kaiming_normal_(m.weight,a=0.2,mode = 'fan_out',nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bais,0)
        elif isinstance(m,nn.BatchNorm2d):
            nn.init.constant_(m.weight,1)
            nn.init.constat_(m.bais,0)

class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.opt=opt
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)

        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)

        self.FFM = FFM(F_g=max(2*N,opt.min_nfc),F_l=max(N,opt.min_nfc),F_int=max(N,opt.min_nfc))

        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)
    
    def forward(self,x):
        x1 = self.head(x)
        x2 = self.body(x1)
        x1 = F.interpolate(x1,size=[x2.size(2),x2.size(3)],mode='bilinear',align_corners=False)
        x3 = self.FFM(g=x1,x=x2)
        x4 = self.tail(x3)
        return x4



class GeneratorConcatSkip2CleanAdd(nn.Module):       # set the  generator model
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.opt=opt
        N = opt.nfc   # super hyper parameters,default=32
        
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)    

        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))         # default  opt.min_nfc = 32
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            #block =ResBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)

        self.FFM = FFM(F_g=max(2*N,opt.min_nfc),F_l=max(N,opt.min_nfc),F_int=max(N,opt.min_nfc))   

        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),3,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),   
            nn.Tanh()
        )

    def forward_once(self,x,y):
        x1 = self.head(x)
        x2 = self.body(x1)
        x1 = F.interpolate(x1,size=[x2.size(2),x2.size(3)],mode='bilinear',align_corners=False)
        x3 = self.FFM(g=x1,x=x2)
        x4 = self.tail(x3)
        ind = int((y.shape[2]-x4.shape[2])/2)   
        y1 = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x4+y1

    def forward(self,x,y):   # x=noise, y=prev
        out1 = self.forward_once(x,y)
        out2 = self.forward_once(x,y)
        
        part1 = out1
        part2 = out2
        criterion = nn.SmoothL1Loss()
        loss = criterion(part1,part2.detach())
        return out1,loss



        

        
