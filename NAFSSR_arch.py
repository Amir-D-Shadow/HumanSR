# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
NAFSSR: Stereo Image Super-Resolution Using NAFNet

@InProceedings{Chu2022NAFSSR,
  author    = {Xiaojie Chu and Liangyu Chen and Wenqing Yu},
  title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
  booktitle = {CVPRW},
  year      = {2022},
}
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from NAFNet_arch import LayerNorm2d, NAFBlock
from arch_util import MySequential
from local_arch import Local_Base

class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=4, stride=4, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=4, stride=4, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=4, stride=4, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=4, stride=4, padding=0)
         
        self.positions = nn.Parameter(torch.randn( 1, c, (32 - 4)// 4 + 1, (32 - 4)// 4 + 1)) # 1, c, h, w
        
        self.up = nn.PixelShuffle(upscale_factor=4)
        self.l_proj3 = nn.Linear(c, c*4*4)
        self.r_proj3 = nn.Linear(c, c*4*4)

    def forward(self, x_l, x_r):
        
        Q_l = self.l_proj1(self.norm_l(x_l)) + self.positions # B1, c, h, w
        h1,w1 = Q_l.shape[-2], Q_l.shape[-1]
        Q_l = einops.rearrange(Q_l.contiguous(),'b c (h) (w) -> b (h w) c') # B1, hxw , c 
        Q_r = self.r_proj1(self.norm_r(x_r)) + self.positions # B2, c, h, w
        h2,w2 = Q_r.shape[-2], Q_r.shape[-1]
        Q_r = einops.rearrange(Q_r.contiguous(),'(b) c (h) (w) -> c (b h w)') # c, B2 x h x w  
        
        V_l = self.l_proj2(x_l) + self.positions # B1, c, h, w
        V_l = einops.rearrange(V_l.contiguous(),'(b) c (h) (w) -> (b h w) c') # B1xhxw , c 
        V_r = self.r_proj2(x_r) + self.positions # B2, c, h, w
        V_r = einops.rearrange(V_r.contiguous(),'(b) c (h) (w) -> (b h w) c') # B2xhxw , c 

        B1 ,seq_len_hxw,c = Q_l.shape
        B2 = x_r.shape[0]
        # (B1, hxw , c) x (c, B2 x h x w) -> (B1, hxw, B2 x h x w)
        attention = torch.matmul(Q_l, Q_r) * self.scale 

        #attention along ref
        F_r2l = torch.matmul(torch.softmax(attention*(B2**(-0.5)), dim=-1), V_r)  #B1, h x w, c
        F_r2l = einops.rearrange(F_r2l.contiguous(), 'b (h w) c -> b h w c',h=h1,w=w1) #B1, h, w, c
        #attention along lr img
        attention = einops.rearrange(attention.contiguous(),'(b1) (m1) (b2 m2) -> b2 m2 (b1 m1)',b2 = B2) #(B2, hxw, B1 x h x w)
        F_l2r = torch.matmul(torch.softmax(attention*(B1**(-0.5)), dim=-1), V_l) #B2, hxw, c
        F_l2r = einops.rearrange(F_l2r.contiguous(), 'b (h w) c -> b h w c',h=h2,w=w2) #B2, h, w, c
        
        #upscale back to original size
        F_r2l = self.l_proj3(F_r2l).permute(0, 3, 1, 2) #B1, c, H, W
        F_r2l = self.up(F_r2l)  #B1, c, H, W
        F_l2r = self.r_proj3(F_l2r).permute(0, 3, 1, 2) #B2, c, H, W
        F_l2r = self.up(F_l2r)     #B2, c, H, W

        # scale
        F_r2l = F_r2l * self.beta
        F_l2r = F_l2r * self.gamma
        
        return x_l + F_r2l, x_r + F_l2r
    
class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x+factor*(new_x-x) for x, new_x in zip(feats, new_feats)])
        return new_feats

class NAFBlockSR(nn.Module):
    '''
    NAFBlock for Super-Resolution
    '''
    def __init__(self, c, fusion=False, drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = SCAM(c) if fusion else None

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats

class NAFNetSR(nn.Module):
    '''
    NAFNet for Super-Resolution
    '''
    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=-1, dual=False):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.body = MySequential(
            *[DropPath(
                drop_path_rate, 
                NAFBlockSR(
                    width, 
                    fusion=(fusion_from <= i and i <= fusion_to), 
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        )
        
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        
        #self.up = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.up_scale = up_scale

    def forward(self, img_lr,ref_img):
        #img_lr: (B1,c,h,w)
        inp_hr = F.interpolate(img_lr, scale_factor=self.up_scale, mode='bilinear') #(B1,c,H,W)
        #ref_img: (B2,c,h,w)
        inp = tuple([img_lr,ref_img]) #[(B1,c,h,w),(B2,c,h,w)]
        feats = [self.intro(x) for x in inp] #[(B1,C,h,w),(B2,C,h,w)]
        feats = self.body(*feats) #[(B1,C,H,W),(B2,C,h,w)]
        #out = torch.cat([self.up(x) for x in feats], dim=1)
        out = self.up(feats[0]) #(B1,c,H,W)
        out = out + inp_hr #(B1,c,H,W)
        return out

class NAFSSR(Local_Base, NAFNetSR):
    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        NAFNetSR.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

if __name__ == '__main__':

    device = torch.device('cuda:5')
    img_lr = torch.randn(5,3,32,32).to(device)
    ref_img = torch.randn(8,3,32,32).to(device)
    net = NAFNetSR(up_scale=8, width=128, num_blks=64, img_channel=3, drop_path_rate=0.1, drop_out_rate=0., fusion_from=-1, fusion_to=1000, dual=True).to(device)
    output = net(img_lr,ref_img)
    print(output.shape)
    print("completed")






