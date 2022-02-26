from PIL import Image
import torch
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx as onnx
import pickle


class SeparableConvBN(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(channels_in, channels_in, kernel_size=3,padding=1, groups=channels_in, bias=False),
            nn.BatchNorm2d(channels_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in, channels_out,kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels_out),
        )

    def forward(self, x):
        return self.blocks(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        a = np.array([1., 2., 1.], dtype=np.float32)
        a2 = a[:, None] * a[None, :]
        filt = torch.tensor(
            a2 / a2.sum())[None, None, :, :].repeat((channels, 1, 1, 1))
        self.register_buffer('filt', filt)
        self.pad = nn.ReflectionPad2d([1, 1, 1, 1])
        self.channels = channels

    def forward(self, x):
        return F.conv2d(self.pad(x), self.filt, stride=2, groups=self.channels)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        blocks = []

        blocks.extend([
            # 64x64
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            Downsample(16),

            # 32x32
            SeparableConvBN(16, 32),
            nn.ReLU(inplace=True),
            Downsample(32),

            # 16x16
            SeparableConvBN(32, 32),
            nn.ReLU(inplace=True),
            Downsample(32),

            # 8x8
            SeparableConvBN(32, 32),
            nn.ReLU(inplace=True),
            Downsample(32),

            # 4x4
            SeparableConvBN(32, 64),
            nn.ReLU(inplace=True),
            Downsample(64),

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Flatten(1),
        ])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


def predict_breed(img_data):
    onnx_model = onnx.load("super_resolution.onnx")
    
