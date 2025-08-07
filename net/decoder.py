from net.basenet import ResBlock, Conv2d_Bn_Relu, DetectNet1
import torch.nn as nn
import torch.nn.functional as F
import torch


class GradPred(nn.Module):
    def __init__(self, channel_list: list = [16, 32, 64, 128, 256]):
        super(GradPred, self).__init__()
        
        self.up_scale = nn.Upsample(scale_factor=2, mode='bilinear')

        self.model_list = nn.ModuleList()

        for i in range(len(channel_list) - 2, -1, -1):
            self.model_list.append(
                Conv2d_Bn_Relu(channel_list[i+1], channel_list[i], kernel_size=3, padding=1),
            )
            self.model_list.append(
                ResBlock(channel_list[i + 1], channel_list[i])
            )

        self.detect = DetectNet1(channel_list[0], 1)

    def forward(self, features: list):
        for i in range(len(features) - 2, -1, -1):
            f_deeper = self.up_scale(features[i+1])
            f_deeper = self.model_list[(len(features) - 2 - i)*2](f_deeper)
            f_fusion = torch.cat([features[i], f_deeper], dim=1)
            f_deeper = self.model_list[(len(features) - 2 - i)*2 + 1](f_fusion)
        res = self.detect(f_deeper)
        return res