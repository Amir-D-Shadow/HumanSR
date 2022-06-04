import torch
import numpy as np
import os
from NAFSSR_arch import NAFNetSR
import torch.nn as nn

class args_parser:
    
    def __init__(self):
        
        #model params
        self.up_scale = 8
        self.width = 128
        self.num_blks = 64
        self.img_channel = 3
        self.drop_path_rate = 0.1
        self.drop_out_rate = 0.0
        self.fusion_from = -1
        self.fusion_to = 1000
        
        #train params
        self.device = torch.device("cuda:5")
        self.device_ids = [5,6,7,8]
        self.iterations = 500
        self.batch_size = 1
        self.ref_size = 48
        
        #misc
        self.load_model = False
        
def train(model):
    
    pass
        

if __name__== "__main__":
    #setting initialize
    opt = args_parser()
    #data loader initialize
    
    #model setup
    net = NAFNetSR(up_scale=opt.up_scale, 
                   width=opt.width, 
                   num_blks=opt.num_blks, 
                   img_channel=opt.img_channel, 
                   drop_path_rate=opt.drop_path_rate, 
                   drop_out_rate=opt.drop_out_rate, 
                   fusion_from=opt.fusion_from, 
                   fusion_to=opt.fusion_from, 
                   dual=True)
    net = nn.DataParallel(net,device_ids=opt.device_ids)
    net.to(opt.device)
    #optimizer set up