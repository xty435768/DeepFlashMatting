import torch
import torch.nn as nn
import torch.nn.functional as F


class TrimapDecoder(nn.Module):
    def __init__(self, feature_channel=256, target_resolution=[64, 128], target_channels=[128, 64], output_channels=3):
        super(TrimapDecoder, self).__init__()
        self.feature_channel = feature_channel
        self.target_resolution = target_resolution
        self.target_channels = target_channels
        self.output_channels = output_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_channel, target_channels[0], kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(target_channels[0]),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(target_channels[0], target_channels[1], kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(target_channels[1]),
            nn.ReLU(),
        )
        self.out_head = nn.Sequential(
            nn.Conv2d(target_channels[1], output_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, size=self.target_resolution[0], mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = F.interpolate(x, size=self.target_resolution[1], mode='bilinear', align_corners=False)
        x = self.out_head(x)
        return x
