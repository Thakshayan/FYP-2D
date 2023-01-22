import torch
import torch.nn as nn
from typing import Dict, Type, Union, Sequence, List


CONV: Dict[int, Union[Type[nn.Conv2d], Type[nn.Conv3d]]] = {2: nn.Conv2d, 3: nn.Conv3d}
DECONV: Dict[int, Union[Type[nn.ConvTranspose2d], Type[nn.ConvTranspose3d]]] = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
DOWN: Dict[int, Union[Type[nn.MaxPool2d], Type[nn.MaxPool3d]]] = {2: nn.MaxPool2d, 3: nn.MaxPool3d}
DROPOUT: Dict[int, Union[Type[nn.Dropout2d], Type[nn.Dropout3d]]] = {2: nn.Dropout2d, 3: nn.Dropout3d}
ACT: Dict[str, Union[Type[nn.ReLU],Type[nn.LeakyReLU],Type[nn.PReLU]]] = {'relu': nn.ReLU, 'leaky': nn.LeakyReLU, 'prelu': nn.PReLU}
NORM: Dict[int, Dict[str,Union[Type[nn.modules.batchnorm._BatchNorm],Type[nn.modules.instancenorm._InstanceNorm],Type[nn.GroupNorm]]]] = {
    2: {
        'batch': nn.BatchNorm2d,
        'instance': nn.InstanceNorm2d,
        'group': nn.GroupNorm,
    },
    3: {
        'batch': nn.BatchNorm3d,
        'instance': nn.InstanceNorm3d,
        'group': nn.GroupNorm,
    }
}


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        dropout: float = 0,
        act: str = "relu",
        norm: str = "batch",
        num_group: int = 6, #8,
        pre_norm: bool = False,
        res: bool = False,
        down: bool = False,
    ) -> None:
        '''Base Block for UNet-Like network
        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dim: 2D or 3D convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides of the input.
            bias: If ``True``, adds a learnable bias to the output.
            dropout: Dropout rate.
            act: Activation function of Block.``relu``,``prelu`` or ``leaky``.
            norm: Normalization type.``batch``,``group`` or ``instance``.
            num_group: number of groups to separate the channels into
            pre_norm: If true, normalization->activation->convolution.
            res: If true, set residual-connection to block.
            down: If true, first conv layer used as downsample layer
        '''
        super().__init__()
        self._check_param(dim, act, norm)
        self.res = res
        stride1 = stride if not down else kernel_size//2+1
        stride2 = stride
        if pre_norm:
            self.conv_x2 = nn.Sequential(
                NORM[dim][norm](num_group, in_channels) if norm == 'group' else NORM[dim][norm](in_channels),
                ACT[act](inplace=True),
                CONV[dim](in_channels, out_channels, kernel_size, stride1, padding, bias=bias),
                DROPOUT[dim](dropout,True),
                NORM[dim][norm](num_group, out_channels) if norm == 'group' else NORM[dim][norm](out_channels),
                ACT[act](inplace=True),
                CONV[dim](out_channels, out_channels, kernel_size, stride2, padding, bias=bias),
                DROPOUT[dim](dropout,True),
            )
        else:
            self.conv_x2 = nn.Sequential(
                CONV[dim](in_channels, out_channels, kernel_size, stride1, padding, bias=bias),
                DROPOUT[dim](dropout,True),
                NORM[dim][norm](num_group, out_channels) if norm == 'group' else NORM[dim][norm](out_channels),
                ACT[act](inplace=True),
                CONV[dim](out_channels, out_channels, kernel_size, stride2, padding, bias=bias),
                DROPOUT[dim](dropout,True),
                NORM[dim][norm](num_group, out_channels) if norm == 'group' else NORM[dim][norm](out_channels),
                ACT[act](inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res:
            return x + self.conv_x2(x)
        return self.conv_x2(x)

    def _check_param(self, dim: int, act: str, norm: str) -> None:
        if dim != 2 and dim != 3:
            raise ValueError(f"Convolution Dim:{dim} is unsupported!")
        if act not in ["relu", "leaky","prelu"]:
            raise ValueError(f"Activation Type:{act} is unsupported!")
        if norm not in ["batch", "instance", "group"]:
            raise ValueError(f"Normalization Type:{norm} is unsupported!")


class UNet_3Plus(nn.Module):
    '''
    UNet 3+: A full-scale connected unet for medical image segmentation
    <https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf>
    '''
    def __init__(
            self,
            in_channels: int=1,
            num_classes: int=1,
            bias: bool = False,
            n_filters: list = [ 64, 128, 256, 512],
            deep_supervision: bool = False
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        C = n_filters[0]*4

        self.encoder1 = Block(in_channels, n_filters[0], 2, bias=bias)
        self.encoder2 = Block(n_filters[0], n_filters[1], 2, bias=bias)
        self.encoder3 = Block(n_filters[1], n_filters[2], 2, bias=bias)
        self.encoder4 = Block(n_filters[2], n_filters[3], 2, bias=bias)
        #self.encoder5 = Block(n_filters[3], n_filters[4], 2, bias=bias)

        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.decoder1 = Block(C, C, 2, bias=bias)
        self.decoder2 = Block(C, C, 2, bias=bias)
        self.decoder3 = Block(C, C, 2, bias=bias)
        self.decoder4 = Block(C, C, 2, bias=bias)
        #self.decoder5 = Block(C, C, 2, bias=bias)

        # Full-Scale Skip Connection
        self.e1_d1 = SkipConv(n_filters[0], n_filters[0], bias=bias)
        self.e1_d2 = SkipConv(n_filters[0], n_filters[0], True, False, 2, bias)
        self.e1_d3 = SkipConv(n_filters[0], n_filters[0], True, False, 4, bias)
        #self.e1_d4 = SkipConv(n_filters[0], n_filters[0], True, False, 8, bias)
        self.e2_d2 = SkipConv(n_filters[1], n_filters[0], bias=bias)
        self.e2_d3 = SkipConv(n_filters[1], n_filters[0], True, False, 2, bias)
        #self.e2_d4 = SkipConv(n_filters[1], n_filters[0], True, False, 4, bias)
        self.e3_d3 = SkipConv(n_filters[2], n_filters[0], bias=bias)
        #self.e3_d4 = SkipConv(n_filters[2], n_filters[0], True, False, 2, bias)
        #self.e4_d4 = SkipConv(n_filters[3], n_filters[0], bias=bias)

        #self.e5_d1 = SkipConv(n_filters[4], n_filters[0], False, True, 16, bias)
        self.e4_d1 = SkipConv(n_filters[3], n_filters[0], False, True, 8, bias)
        self.e4_d2 = SkipConv(n_filters[3], n_filters[0], False, True, 4, bias)
        self.e4_d3 = SkipConv(n_filters[3], n_filters[0], False, True, 2, bias)

        #self.d4_d1 = SkipConv(C, n_filters[0], False, True, 8, bias)
        #self.d4_d2 = SkipConv(C, n_filters[0], False, True, 4, bias)
        #self.d4_d3 = SkipConv(C, n_filters[0], False, True, 2, bias)

        self.d3_d1 = SkipConv(C, n_filters[0], False, True, 4, bias)
        self.d3_d2 = SkipConv(C, n_filters[0], False, True, 2, bias)

        self.d2_d1 = SkipConv(C, n_filters[0], False, True, 2, bias)

        # Output Conv Layer in the official code uses 3 as kernel_size
        if deep_supervision:
            self.out1 = nn.Conv2d(C, num_classes, 3, 1, 1, bias=bias)
            self.out2 = nn.Conv2d(C, num_classes, 3, 1, 1, bias=bias)
            self.out3 = nn.Conv2d(C, num_classes, 3, 1, 1, bias=bias)
            #self.out4 = nn.Conv2d(C, num_classes, 3, 1, 1, bias=bias)
        else:
            self.out = nn.Conv2d(C, num_classes, 3, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        #e5 = self.encoder5(self.pool(e4))

        # Fuse Decoder 4
        #t1 = self.e1_d4(e1)
        #t2 = self.e2_d4(e2)
        #t3 = self.e3_d4(e3)
        #t4 = self.e4_d4(e4)
        #t5 = self.e5_d4(e5)
        #fusion = torch.cat([t1, t2, t3, t4, t5], dim=1)
        #d4 = self.decoder4(fusion)

        # Fuse Decoder 3
        t1 = self.e1_d3(e1)
        t2 = self.e2_d3(e2)
        t3 = self.e3_d3(e3)
        t4 = self.e4_d3(e4)
        #t5 = self.e5_d3(e5)
        fusion = torch.cat([t1, t2, t3, t4], dim=1)
        d3 = self.decoder3(fusion)

        # Fuse Decoder 2
        t1 = self.e1_d2(e1)
        t2 = self.e2_d2(e2)
        t3 = self.e4_d2(e4)
        t4 = self.d3_d2(d3)
        # t3 = self.d3_d2(d3)
        # t4 = self.d4_d2(d4)
        # t5 = self.e5_d2(e5)
        fusion = torch.cat([t1, t2, t3, t4], dim=1)
        d2 = self.decoder4(fusion)

        # Fuse Decoder 1
        t1 = self.e1_d1(e1)
        t2 = self.d2_d1(d2)
        t3 = self.d3_d1(d3)
        #t4 = self.d4_d1(d4)
        t4 = self.e4_d1(e4)
        fusion = torch.cat([t1, t2, t3, t4], dim=1)
        d1 = self.decoder1(fusion)

        if self.deep_supervision:
            out1 = self.out1(d1)
            out2 = self.out2(d2)
            out3 = self.out3(d3)
            
            return [out1, out2, out3]
        else:
            return self.out(d1)


class SkipConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = False,
        up: bool = False,
        scale_factor: int = 2,
        bias: bool = False,
    ) -> None:
        '''Full-scale Skip Connections in the paper'''
        super().__init__()
        assert (pool and up) == False, "Skip connection should be only downsampling or upsampling"
        self.pool = pool
        self.up = up
        if pool:
            self.downsample = nn.MaxPool2d(scale_factor, scale_factor, ceil_mode=True)
        if up:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool:
            x = self.downsample(x)
        elif self.up:
            x = self.upsample(x)
        x = self.relu(self.bn(self.conv(x)))
        return x