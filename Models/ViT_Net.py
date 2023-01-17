import torch
import torch.nn as nn
from ViT import ViT3D

class ViT_Net(nn.Module):
    def __init__(self):
        super(ViT_Net, self).__init__()
        
        # Encoder 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        # Encoder 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        # Encoder 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        
        # Encoder 4
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)

        # Decoder 1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv9 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Decoder 2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Decoder 3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv13 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        #Transformer
        self.transformer = ViT3D(embedding_dim=256, head_num=4, block_num=8)
       
        
        # Output layer
        self.conv15 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder 1
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1_pool = self.pool1(x1)

        # Encoder 2
        x2 = self.conv3(x1_pool)
        x2 = self.conv4(x2)
        x2_pool = self.pool2(x2)
        
        # Encoder 3
        x3 = self.conv5(x2_pool)
        x3 = self.conv6(x3)
        x3_pool = self.pool3(x3)
        
        # Encoder 4
        x4 = self.conv7(x3_pool)
        x4 = self.conv8(x4)
        x4_pool = self.pool4(x4)
        
        # Decoder 1
        x5 = self.up1(x4_pool)
        x3_transfrom = self.transformer(x3)
        x5 = torch.cat((x5, x3_transfrom), dim=1)
        x5 = self.conv9(x5)
        x5 = self.conv10(x5)
        
        # Decoder 2
        x6 = self.up2(x5)
        x2_transfrom = self.transformer(x2)
        x6 = torch.cat((x6, x2_transfrom), dim=1)
        x6 = self.conv11(x6)
        x6 = self.conv12(x6)
        
        # Decoder 3
        x7 = self.up3(x6)
        x1_transfrom = self.transformer(x1)
        x7 = torch.cat((x7, x1_transfrom), dim=1)
        x7 = self.conv13(x7)
        x7 = self.conv14(x7)

        # Output layer
        x8 = self.conv15(x7)

        return x8
