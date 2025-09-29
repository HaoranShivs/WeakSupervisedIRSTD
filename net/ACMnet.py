import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss import SoftLoULoss_Epochs, SoftLoULoss


class _FCNHead(nn.Module):
    """
    PyTorch version of GluonCV's _FCNHead.
    A simple FCN head for semantic segmentation.
    Typically consists of:
        - Conv (kernel_size=3, padding=1) + Norm + ReLU
        - Dropout (optional, commonly used)
        - Conv (kernel_size=1) to map to num_classes
    """

    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, dropout=0.1):
        """
        Args:
            in_channels (int): Number of input channels.
            channels (int): Number of intermediate feature channels.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.BatchNorm2d.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(_FCNHead, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=1)  # Output channels = num_classes
        )

    def forward(self, x):
        return self.block(x)


class CIFARBasicBlockV1(nn.Module):
    """
    PyTorch version of GluonCV's CIFARBasicBlockV1.
    A basic residual block for ResNet on CIFAR datasets.
    """

    def __init__(self, channels, stride=1, downsample=False, in_channels=0, norm_layer=nn.BatchNorm2d):
        """
        Args:
            channels (int): Number of output channels for the block.
            stride (int, optional): Stride for the first convolutional layer. Defaults to 1.
            downsample (bool, optional): Whether to apply downsampling (via stride or 1x1 conv) in the shortcut path. Defaults to False.
            in_channels (int, optional): Number of input channels. If 0, assumed to be equal to `channels`. Defaults to 0.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.BatchNorm2d.
        """
        super(CIFARBasicBlockV1, self).__init__()

        self.channels = channels
        self.stride = stride
        self.downsample = downsample
        self.in_channels = in_channels if in_channels != 0 else channels

        # Main path
        self.conv1 = nn.Conv2d(self.in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(channels)

        # Shortcut path (identity or 1x1 conv for dimension matching)
        if downsample:
            self.downsample_conv = nn.Conv2d(self.in_channels, channels, kernel_size=1, stride=stride, bias=False)
            self.downsample_bn = norm_layer(channels)
        else:
            self.downsample_conv = None
            self.downsample_bn = None

        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply shortcut if needed
        if self.downsample:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity)

        out += identity
        out = self.relu_out(out)

        return out


class AsymBiChaFuse(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)

        # Top-down pathway: from high-level feature (xh)
        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Conv2d(channels, self.bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottleneck_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # Bottom-up pathway: from low-level feature (xl)
        self.bottomup = nn.Sequential(
            nn.Conv2d(channels, self.bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottleneck_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # Post-processing
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, xh, xl):
        """
        Args:
            xh (Tensor): High-level feature map (e.g., from deeper layer)
            xl (Tensor): Low-level feature map (e.g., from shallower layer)

        Returns:
            Tensor: Fused feature map
        """
        # Generate attention weights
        topdown_wei = self.topdown(xh)      # [B, C, 1, 1] → broadcast to [B, C, H, W]
        bottomup_wei = self.bottomup(xl)    # [B, C, H, W]

        # Fuse: asymmetric bi-directional attention
        # MXNet: 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = 2 * xl * topdown_wei + 2 * xh * bottomup_wei

        # Post-process
        xs = self.post(xs)

        return xs


class ASKCResUNet(nn.Module):
    def __init__(self, layers, channels, fuse_mode, tiny=True, classes=1,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(ASKCResUNet, self).__init__()

        self.layer_num = len(layers)
        self.tiny = tiny

        stem_width = int(channels[0])
        self.stem = nn.Sequential()
        self.stem.add_module('norm0', norm_layer(3))

        if tiny:
            self.stem.add_module('conv1', nn.Conv2d(3, stem_width * 2, kernel_size=3, stride=1,
                                                    padding=1, bias=False))
            self.stem.add_module('norm1', norm_layer(stem_width * 2))
            self.stem.add_module('relu1', nn.ReLU(inplace=True))
        else:
            self.stem.add_module('conv1', nn.Conv2d(3, stem_width, kernel_size=3, stride=2,
                                                    padding=1, bias=False))
            self.stem.add_module('norm1', norm_layer(stem_width))
            self.stem.add_module('relu1', nn.ReLU(inplace=True))
            self.stem.add_module('conv2', nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                                                    padding=1, bias=False))
            self.stem.add_module('norm2', norm_layer(stem_width))
            self.stem.add_module('relu2', nn.ReLU(inplace=True))
            self.stem.add_module('conv3', nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                                                    padding=1, bias=False))
            self.stem.add_module('norm3', norm_layer(stem_width * 2))
            self.stem.add_module('relu3', nn.ReLU(inplace=True))
            self.stem.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[0],
                                       channels=channels[1], stride=1, stage_index=1,
                                       in_channels=channels[1], norm_layer=norm_layer)

        self.layer2 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[1],
                                       channels=channels[2], stride=2, stage_index=2,
                                       in_channels=channels[1], norm_layer=norm_layer)

        self.layer3 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[2],
                                       channels=channels[3], stride=2, stage_index=3,
                                       in_channels=channels[2], norm_layer=norm_layer)

        self.deconv2 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=4,
                                          stride=2, padding=1)
        self.uplayer2 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[1],
                                         channels=channels[2], stride=1, stage_index=4,
                                         in_channels=channels[2], norm_layer=norm_layer)
        self.fuse2 = self._fuse_layer(fuse_mode, channels=channels[2])

        self.deconv1 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=4,
                                          stride=2, padding=1)
        self.uplayer1 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[0],
                                         channels=channels[1], stride=1, stage_index=5,
                                         in_channels=channels[1], norm_layer=norm_layer)
        self.fuse1 = self._fuse_layer(fuse_mode, channels=channels[1])

        self.head = _FCNHead(in_channels=channels[1], channels=classes)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0, norm_layer=nn.BatchNorm2d):
        downsample = (channels != in_channels) or (stride != 1)
        layers_list = [
            block(channels, stride, downsample, in_channels=in_channels,
                  norm_layer=norm_layer)
        ]
        for _ in range(1, layers):
            layers_list.append(
                block(channels, 1, False, in_channels=channels, norm_layer=norm_layer)
            )
        return nn.Sequential(*layers_list)

    def _fuse_layer(self, fuse_mode, channels):
        # if fuse_mode == 'DirectAdd':
        #     return DirectAddFuse(channels=channels)
        # elif fuse_mode == 'Concat':
        #     return ConcatFuse(channels=channels)
        # elif fuse_mode == 'SK':
        #     return SKFuse(channels=channels)
        # elif fuse_mode == 'BiLocal':
        #     return BiLocalChaFuse(channels=channels)
        # elif fuse_mode == 'BiGlobal':
        #     return BiGlobalChaFuse(channels=channels)
        if fuse_mode == 'AsymBi':
            return AsymBiChaFuse(channels=channels)
        # elif fuse_mode == 'TopDownGlobal':
        #     return TopDownGlobalChaFuse(channels=channels)
        # elif fuse_mode == 'TopDownLocal':
        #     return TopDownLocalChaFuse(channels=channels)
        else:
            raise ValueError(f'Unknown fuse_mode: {fuse_mode}')

    def forward(self, x):
        _, _, hei, wid = x.shape

        x = self.stem(x)      # e.g., 480x480 → 16 channels
        c1 = self.layer1(x)   # 480x480, 16
        c2 = self.layer2(c1)  # 240x240, 32
        c3 = self.layer3(c2)  # 120x120, 64

        deconvc2 = self.deconv2(c3)       # 240x240, 32
        fusec2 = self.fuse2(deconvc2, c2) # 240x240, 32
        upc2 = self.uplayer2(fusec2)      # 240x240, 32

        deconvc1 = self.deconv1(upc2)     # 480x480, 16
        fusec1 = self.fuse1(deconvc1, c1) # 480x480, 16
        upc1 = self.uplayer1(fusec1)      # 480x480, 16

        pred = self.head(upc1)            # [B, classes, H, W]

        if self.tiny:
            out = pred
        else:
            out = F.interpolate(pred, size=(hei, wid), mode='bilinear', align_corners=False)

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)
    

class ASKCResUNet_withloss(nn.Module):
    def __init__(self, epoch_ratio=0.75):
        super(ASKCResUNet_withloss, self).__init__()

        layers = [4] * 3
        channels = [x * 1 for x in [8, 16, 32, 64]]
        tiny = True
        classes=1
        norm_layer=nn.BatchNorm2d

        self.net = ASKCResUNet(layers, channels, 'AsymBi', tiny, classes, norm_layer)
        self.softiou_loss_fn = SoftLoULoss_Epochs(epoch_ratio)
        # self.softiou_loss_fn = SoftLoULoss()

    def forward(self, img, label, curr_epoch_ratio=0):
        img = img.repeat(1, 3, 1, 1)  # for DNANet
        res = self.net(img)
        pred = F.sigmoid(res)
        # print("res min/max/mean:", res.min().item(), res.max().item(), res.mean().item())
        # print("pred min/max/mean:", pred.min().item(), pred.max().item(), pred.mean().item())
        loss = self.softiou_loss_fn(pred, label, curr_epoch_ratio)
        return pred, loss

