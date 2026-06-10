"""
Unet.py
This file contains the implementation of the Unet model for image segmentation tasks. 
The Unet architecture consists of a contracting path (encoder) and an expansive path (decoder) with skip connections between corresponding layers. 
The model is designed to capture both local and global features of the input image, making it effective for tasks such as medical image segmentation.
It includes the following components:
- ConvdBlock: A convolutional block that consists of a convolutional layer, batch normalization, and a ReLU activation function.
- Unet: The main Unet model that defines the architecture of the encoder and decoder paths, as well as the forward pass through the network.
        The model takes an input image and produces a segmented output image, where each pixel is classified into one of the target classes.
        The architecture includes downsampling layers for the encoder, upsampling layers for the decoder,
        and skip connections to preserve spatial information from the encoder to the decoder.

Author: Ladipo Ipadeola
Date: 06/10/2026

"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvdBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel=3, stride=1, padding=1):
        super.__init__()
        self.Convd_layer = nn.Conv2d(
            in_channels=in_size, out_channels=out_size, kernel_size=kernel, padding=padding, stride=stride)
        self.Relu_layer = nn.modules.activation.ReLU(inplace=True)
        self.BN_layer = nn.BatchNorm2d(out_size)

    def forward(self, inputs):
        outputs = self.Convd_layer(inputs)
        outputs = self.BN_layer(outputs)
        outputs = self.Relu_layer(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self):
        super.__init__()
        self.down_1 = nn.Sequential(ConvdBlock(
            in_size=3, out_size=16), ConvdBlock(in_size=16, out_size=32, stride=2))
        self.down_2 = nn.Sequential(ConvdBlock(
            in_size=32, out_size=64), ConvdBlock(in_size=64, out_size=128))
        self.middle = ConvdBlock(
            in_size=128, out_size=128, kernel=1, padding=0)
        self.up_1 = nn.Sequential(ConvdBlock(
            in_size=256, out_size=128), ConvdBlock(in_size=128, out_size=32))
        self.up_2 = nn.Sequential(ConvdBlock(
            in_size=64, out_size=64), ConvdBlock(in_size=64, out_size=32))
        self.output_layer = nn.Sequential(ConvdBlock(in_size=32, out_size=16), ConvdBlock(
            in_size=16, out_size=1, kernel=1, padding=0))
        self.maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        skip_1 = self.down_1(inputs)
        outs = self.maxpool_layer(skip_1)
        skip_2 = self.down_2(outs)
        outs = self.maxpool_layer(outs)
        outs = self.middle(outs)
        outs = F.interpolate(outs, scale_factor=2)
        outs = torch.concat([skip_2, outs], dim=1)
        outs = self.up_1(outs)
        outs = F.interpolate(outs, scale_factor=2)
        outs = torch.concat([skip_1, outs], dim=1)
        outs = self.up_2(outs)
        outs = F.interpolate(outs, scale_factor=2)
        results = self.output_layer(outs)
        return results
