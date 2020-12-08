#-*-coding:utf-8-*
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_upsample_layer, constant_init,
                      normal_init)

from ..registry import HEADS

DUCs = [256, 128]


@HEADS.register_module()
class TopDown1DUC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TopDown1DUC, self).__init__()
        DUCs = [int(in_channels/2), int(in_channels/4)]
        # print(setting)
        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(DUCs[1], DUCs[0], upscale_factor=2)
        self.conv_out = nn.Conv2d(
            int(DUCs[0]/4), out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.suffle1(x)
        out = self.duc1(out)
        out = self.conv_out(out)
        return out

    def init_weights(self):
        pass
        """Initialize model weights."""
        # for _, m in self.deconv_layers.named_modules():
        #     if isinstance(m, nn.ConvTranspose2d):
        #         normal_init(m, std=0.001)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         constant_init(m, 1)
        # for m in self.final_layer.modules():
        #     if isinstance(m, nn.Conv2d):
        #         normal_init(m, std=0.001, bias=0)


class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x
