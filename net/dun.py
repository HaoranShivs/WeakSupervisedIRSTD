import torch
import torch.nn as nn
from basenet import ResBlock, Conv2d_Bn_Relu
from utils.loss import SoftLoULoss


class upscaler_linear(nn.Module):
    """
    Last part of BEM
    """
    def __init__(self, outchannels):
        super(upscaler_linear, self).__init__()

        self.l4 = Conv2d_Bn_Relu(outchannels[3], outchannels[2], 1, 1)
        self.l3 = Conv2d_Bn_Relu(outchannels[2], outchannels[1], 1, 1)
        self.l2 = Conv2d_Bn_Relu(outchannels[1], outchannels[0], 1, 1)
        self.l1 = nn.Conv2d(outchannels[0], 1, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs

        rec_x4 = self.l4(x4).repeat(1,1,2,2)
        rec_x3 = self.l3(x3 + rec_x4).repeat(1,1,2,2)
        rec_x2 = self.l2(x2 + rec_x3).repeat(1,1,2,2)
        rec_x1 = self.sigmoid(self.l1(x1 + rec_x2)).repeat(1,1,2,2)
        
        return rec_x1


class BEM(nn.Module):
    def __init__(self, outchannels, downsampler):
        super(BEM, self).__init__()
        self.downsampler = downsampler

        self.resb1 = ResBlock(1, outchannels[0], outchannels[0], 3, 1)
        self.resb2 = ResBlock(outchannels[0], outchannels[1], outchannels[1], 3, 1)
        self.resb3 = ResBlock(outchannels[1], outchannels[2], outchannels[2], 3, 1)
        self.resb4 = ResBlock(outchannels[2], outchannels[3], outchannels[3], 3, 1)

        # self.convT4, self.bn4 = nn.ConvTranspose2d(outchannels[3], outchannels[2], 3, 2, 1, 1), nn.BatchNorm2d(outchannels[2])
        # self.convT3, self.bn3 = nn.ConvTranspose2d(outchannels[2], outchannels[1], 3, 2, 1, 1), nn.BatchNorm2d(outchannels[1])
        # self.convT2, self.bn2 = nn.ConvTranspose2d(outchannels[1], outchannels[0], 3, 2, 1, 1), nn.BatchNorm2d(outchannels[0])
        # self.convT1 = nn.ConvTranspose2d(outchannels[1], 1, 3, 2, 1, 1)

        self.upscaler = upscaler_linear(outchannels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x1 = self.downsampler(self.resb1(inputs))   #(B, C, 128, 128)
        x2 = self.downsampler(self.resb2(x1))
        x3 = self.downsampler(self.resb3(x2))
        x4 = self.downsampler(self.resb4(x3))   #(B, C, 16, 16)
        
        B = self.upscaler((x1, x2, x3, x4)) #(B, 1, 256, 256)
        return B


class TEM(nn.Module):
    def __init__(self, hiden_channel:int=32):
        super(TEM, self).__init__()

        self.conv1 = nn.Conv2d(1, hiden_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(hiden_channel, hiden_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(hiden_channel, 1, 3, 1, 1)

        self.relu = nn.ReLU()
        self.attentioner = nn.Sigmoid()

    def forward(self, inputs):
        x = self.relu(self.conv1(inputs))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        attention = self.attentioner(x)

        T = attention * inputs
        
        return T


class DUN2net(nn.Module):
    def __init__(self, cfg=None):
        super(TEM, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")
        
        downsampler = nn.MaxPool2d(2, 2)
        self.bem1 = BEM(cfg["multiscalefeature_outchannel"], downsampler)
        self.tem1 = TEM(cfg["hide_channel"])

        self.bem2 = BEM(cfg["multiscalefeature_outchannel"], downsampler)
        self.tem2 = TEM(cfg["hide_channel"])
    
    def __init__(self, inputs):
        B1 = self.bem1(inputs)
        TN1 = inputs - B1
        T1 = self.tem1(TN1)
        D1 = B1 + T1
        B2 = self.bem2(D1)
        TN2 = D1 - B2
        T2 = self.tem2(TN2)
        D2 = B2 + T2
        return (D1, T1), (D2, T2)


class recons_loss(nn.Module):
    def __init__(self, cfg=None):
        super(recons_loss, self).__init__()

    def forward(self, origin, D):
        """
        inputs:
            origin, D: (N/B, B, 1, 256, 256)
        outputs:
            loss: (N/B,)
        """
        diff = origin - D
        return torch.sum(diff * diff, dim=(-1,-2,-3,-4))
    

class total_loss(nn.Module):
    def __init__(self, cfg=None):
        super(total_loss, self).__init__()
        self.ratio = cfg["ratio_list"]

        self.recons_loss = recons_loss()
        self.softIoU = SoftLoULoss()

    def forward(self, imgs, D, masks, T):
        """
        inputs:
            imgs, masks: (B, 1, 256, 256)
            D, T: (B*n, 1, 256, 256)
        outputs:
            loss: (1,)
        """
        B, _, _, _ = imgs.shape
        N, _, _, _ = D.shape
        # imgs = imgs.unsqueeze(0).repeat(N/B, 1, 1, 1, 1)
        # D = D.unsqueeze(0).view(N/B, B, *(D.shape[2:]))
        loss = torch.tensor(0.0)
        for i in range(N/B):
            loss = loss + self.ratio[i] * (recons_loss(imgs, D[i*B:(i+1)*B]) + self.softIoU(D[i*B:(i+1)*B], masks))

        return loss