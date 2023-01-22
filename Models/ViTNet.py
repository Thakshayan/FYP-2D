import torch
import torch.nn as nn
from .ViT import ViT

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class ViTNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64,1, 1)

        #Transformer
        self.transformer1 = ViT(img_dim=32, in_channels=256, embedding_dim=256, block_num=1 )
        self.transformer2 = ViT(img_dim=64, in_channels=128, embedding_dim=128, block_num=1 )
        self.transformer3 = ViT(img_dim=128, in_channels=64, embedding_dim=64,block_num=1 )
        
        
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)
        conv_3 = self.transformer1(conv3)  
        print(conv_3.shape) 
        x = torch.cat([x, conv_3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x) 
        conv_2 = self.transformer2(conv2)  
        print(conv_2.shape)        
        x = torch.cat([x, conv_2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)  
        conv_1 = self.transformer3(conv1)  
        x = torch.cat([x, conv_1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out