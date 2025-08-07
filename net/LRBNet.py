import torch
import torch.nn as nn


class LRBENet(nn.Module):
    def __init__(self, channel=32, alpha=0.5):
        super(LRBENet, self).__init__()
        self.alpha = alpha
        self.channel = channel

        self.conv1 = nn.Conv2d(1, channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(channel)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(True)

    def forward(self, img):
        """
        args:
            img(torch.tensor): (B, C, H, W)
        output:
            reconstructed_img(torch.tensor): (B, C, H, W)
        """
        recons_img = torch.zeros_like(img)
        x1 = self.maxpool(self.relu(self.bn1(self.conv1(img))))  # (B, 32, 64, 64)
        recons_img += self.alpha * x1.repeat(1, 1, 4, 4) + (
            1 - self.alpha
        ) * img.repeat(1, 32, 1, 1)
        x2 = self.maxpool(self.relu(self.bn2(self.conv2(x1))))  # (B, 32, 16, 16)
        recons_img += (
            self.alpha * x2.repeat(1, 1, 4 * 4, 4 * 4) + (1 - self.alpha) * recons_img
        )
        x3 = self.maxpool(self.relu(self.bn3(self.conv3(x2))))  # (B, 32, 4, 4)
        recons_img += (
            self.alpha * x3.repeat(1, 1, 4 * 4 * 4, 4 * 4 * 4)
            + (1 - self.alpha) * recons_img
        )
        recons_img = torch.sum(recons_img, 1, True) / self.channel
        return recons_img


class TENet(nn.Module):
    """
    extract the target object from
    """

    def __init__(self, channel=32, alpha=0.5):
        super(LRBENet, self).__init__()
        self.alpha = alpha

        self.conv1 = nn.Conv2d(1, channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(channel)
        self.conv_recons = nn.Conv2d(channel, 1, 1, 1, 1)

        self.maxpool = nn.MaxPool2d(4, 4)
        self.relu = nn.ReLU(True)

    def forward(self, img):
        x = img
        return x
