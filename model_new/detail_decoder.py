import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Callable, Optional
from einops import rearrange
import os

import torchvision.transforms as T

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def convT(in_planes: int, out_planes: int):
    return nn.ConvTranspose2d(in_planes, 
                              out_planes, 
                              kernel_size=4, 
                              stride=2, 
                              padding=1, 
                              bias=False)

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, in_channels, channels, dims=2):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        self.op = conv3x3(in_channels, channels, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.in_channels
        return self.op(x)

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, in_channels, channels, use_conv, dims=2):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        self.conv = conv3x3(in_channels, channels)

    def forward(self, x):
        assert x.shape[1] == self.in_channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="bilinear"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
        if self.use_conv:
            x = self.conv(x)
        return x

class BasicDownBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        downsample = None,
        adjust_outsize_to_residualsize=False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.downsample = downsample
        self.adjust_outsize_to_residualsize = adjust_outsize_to_residualsize

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None:
            identity = self.downsample(x)
        
        if self.adjust_outsize_to_residualsize:
            out = F.interpolate(out, size=identity.shape[-2:], mode='bilinear', align_corners=False)
        out += identity
        out = self.relu(out)

        return out


class BasicUpBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        upsample = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = convT(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample != None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class Matting_Head(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """
    def __init__(
        self,
        in_chans = 32,
        mid_chans = 16,
        out_chans = 1,
    ):
        super().__init__()
        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, 3, 1, 1),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(True),
            nn.Conv2d(mid_chans, out_chans, 1, 1, 0)
            )

    def forward(self, x):
        x = self.matting_convs(x)

        return x

class MattingDecoder(nn.Module):
    def __init__(
        self,
        downsample_in = [6, 32, 64, 128],
        upsample_in = [256, 128, 128, 64, 32],
    ):
        super().__init__()
        assert len(downsample_in) == len(upsample_in) - 1
        self.down_blks = nn.ModuleList()
        for i in range(len(downsample_in) - 1):
            downsample = Downsample(
                in_channels=downsample_in[i] if i == 0 else downsample_in[i], # + vpd_features_channel_out,
                channels=downsample_in[i+1]
            )
            self.down_blks.append(
                BasicDownBlock(
                    inplanes=downsample_in[i] if i == 0 else downsample_in[i], # + vpd_features_channel_out,
                    planes=downsample_in[i+1],
                    stride=2,
                    downsample=downsample,
                )
            )
        self.mid_blk = BasicDownBlock(
            inplanes=upsample_in[0],
            planes=upsample_in[1],
            downsample=nn.Sequential(
                nn.Conv2d(upsample_in[0], upsample_in[1], kernel_size=3, stride=1, padding=1)
            ),
            adjust_outsize_to_residualsize=True,
        )
        self.up_blks = nn.ModuleList()
        for i in range(1, len(upsample_in) - 1):
            upsample = Upsample(
                in_channels=upsample_in[i] + downsample_in[-i],
                channels=upsample_in[i+1],
                use_conv=True
            )
            self.up_blks.append(
                BasicUpBlock(
                    inplanes=upsample_in[i] + downsample_in[-i],
                    planes=upsample_in[i+1],
                    upsample=upsample,
                )
            )

        self.alpha_head = Matting_Head(
            in_chans = upsample_in[-1],
            out_chans=1, # alpha
        )
        self.fg_head = Matting_Head(
            in_chans = upsample_in[-1],
            out_chans=3, # foreground residual
        )

    def forward(self, img, trimap_logit, backbone_features):
        if trimap_logit.shape[-2:] != img.shape[-2:]:
            trimap_logit = F.interpolate(trimap_logit, size=img.shape[-2:], mode='bilinear', align_corners=False)
        trimap = F.one_hot(torch.argmax(trimap_logit, dim=1), num_classes=3).permute(0, 3, 1, 2).float()
        details = []
        dfeatures = torch.cat([img, trimap], dim=1) # img 0,1 trimap 0,1
        for i in range(len(self.down_blks)):
            dfeatures = self.down_blks[i](dfeatures)
            details.append(dfeatures)
        hfeatures = self.mid_blk(backbone_features)
        for i in range(len(self.up_blks)):
            if hfeatures.shape[-2:] != details[-(i+1)].shape[-2:]:
                hfeatures = F.interpolate(hfeatures, size=details[-(i+1)].shape[-2:], mode='bilinear', align_corners=False)
            hfeatures = torch.cat([hfeatures, details[-(i+1)]], dim=1)
            hfeatures = self.up_blks[i](hfeatures)
        
        pred_alpha = self.alpha_head(hfeatures)
        pred_alpha = torch.clamp(pred_alpha, 0., 1.)
        pred_fg_raw = self.fg_head(hfeatures)
        pred_fg = torch.clamp(pred_fg_raw + img, 0., 1.)
        pred_hidden = nn.ReLU()(hfeatures)
        return {
            'alpha': pred_alpha,
            'fg': pred_fg,
            'fg_raw': pred_fg_raw,
            'hidden': pred_hidden
        }

if __name__ == '__main__':
    device = 'cuda:0'
    model = MattingDecoder(
        model_channels=32,
        emb_channels=32
    ).to(device)
    features = torch.rand(7, 384, 32, 32).to(device)
    inputs = torch.rand(4, 4, 512, 512).to(device)
    timesteps = torch.randint(low=0, high=200, size=(4, )).to(device)

    out = model(features, inputs, timesteps)
    phas = out['phas']
    print(phas.shape)

