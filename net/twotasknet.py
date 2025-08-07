import torch
import torch.nn as nn

# # 设置打印选项
# torch.set_printoptions(
#     threshold=1024,         # 最大显示元素数量为10
#     linewidth=120,        # 每行的最大字符数为120
#     precision=4,          # 小数点后的数字精度为4
#     # edgeitems=5,          # 每行显示的边缘元素数量为5
#     sci_mode=False        # 不使用科学计数法
# )

from net.basenet import Resconv, ShallowFeatureExtractor, DeepFeatureExtractor, UpScaler, Conv2d_Bn_Relu, DetectNet1
from utils.loss import SoftLoULoss, Heatmap_SoftIoU, Heatmap_MSE

import matplotlib.pyplot as plt
import numpy as np


class MultiScaleFeatureNet(nn.Module):
    def __init__(self, in_channel, out_channel_list, downsampler=None):
        super(MultiScaleFeatureNet, self).__init__()

        out_channel_list = [in_channel] + out_channel_list

        pyramid_layers = []
        for i in range(1, len(out_channel_list)):
            layer = ShallowFeatureExtractor(out_channel_list[i - 1], out_channel_list[i], 3, 1, downsampler)
            pyramid_layers.append(layer)

        self.pyramid = nn.ModuleList(pyramid_layers)

    def forward(self, shallow_feature):
        """
        args:
            shallow_feature(torch.tensor): (B, C1, H, W)
        output:
            (...,H/2, W/2)
            (...,H/4, W/4)
            (...,H/8, W/8)
            ...
        """
        outputs = []
        x = shallow_feature
        for i in self.pyramid:
            x = i(x)
            outputs.append(x)
        return outputs


class FusionNet_upscale(nn.Module):
    def __init__(self, in_channel_list):
        super(FusionNet_upscale, self).__init__()

        SRer_layers = []
        for i in range(len(in_channel_list) - 1, 0, -1):
            # SRer_layers.append(UpScaler(in_channel_list[i], in_channel_list[i - 1], 3))
            SRer_layers.append(ConvBlockUS(in_channel_list[i], in_channel_list[i - 1]))

        self.SRer = nn.ModuleList(SRer_layers)

    def forward(self, inputs):
        if len(inputs) != len(self.SRer) + 1:
            print(len(inputs), len(self.SRer))
            raise ValueError("inputs of FusionNet is not same with cfg['multiscalefeature_outchannel']")

        outputs = []
        xd = inputs[len(inputs) - 1]
        for i in range(0, len(self.SRer)):
            xs = inputs[len(inputs) - 2 - i]
            xd = torch.max(self.SRer[i](xd), xs) 
            outputs.append(xd)
        return outputs


class Heatmap(nn.Module):
    def __init__(self, in_channel_list):
        super(Heatmap, self).__init__()

        heatmaps = []
        for i in range(len(in_channel_list) - 1, -1, -1):
            heatmaps.append(
                nn.Sequential(
                    DeepFeatureExtractor(in_channel_list[i], in_channel_list[i], 3),
                    nn.Conv2d(in_channel_list[i], 1, 1),
                )
            )

        self.heatmaps = nn.ModuleList(heatmaps)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        if len(inputs) != len(self.heatmaps):
            raise ValueError("inputs of FusionNet is not same with cfg['multiscalefeature_outchannel']")

        outputs = []
        for i in range(0, len(self.heatmaps)):
            x = inputs[i]
            res = self.heatmaps[i](x)
            res = self.sigmoid(res)
            outputs.append(res)
        return outputs


class Heatmap_net(nn.Module):
    """
    args:
        in_channel:
        cfg: dict of configuration information
    inputs:
        img: (B, 1, H, W)
    output:
        (B, 1, 8, 8)
        (B, 1, 16, 16)
        (B, 1, 32, 32)
    """

    def __init__(self, cfg=None):
        super(Heatmap_net, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler1 = nn.MaxPool2d(3, 2, 1)
        self.resconv = Resconv(1, cfg["resconv_outchannel"], downsampler1)

        downsampler2 = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"],
            cfg["multiscalefeature_outchannel"],
            downsampler2,
        )
        self.ffusion = FusionNet_upscale(cfg["multiscalefeature_outchannel"][-4:])
        self.heatmap = Heatmap(cfg["multiscalefeature_outchannel"][-4:-1])

    def forward(self, img):
        x = self.resconv(img)
        outputs_f = self.multiscalef(x)
        xf1 = self.ffusion(outputs_f[-4:])  # (B, 32, 256, 256)
        target = self.heatmap(xf1)
        return target


class Heatmap_net2(nn.Module):
    """
    args:
        in_channel:
        cfg: dict of configuration information
    inputs:
        img: (B, 1, H, W)
    output:
        (B, 1, 8, 8)
        (B, 1, 16, 16)
        (B, 1, 32, 32)
    """

    def __init__(self, cfg=None):
        super(Heatmap_net2, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        self.resconv = ConvBlockDS(1, cfg["resconv_outchannel"])

        downsampler2 = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"],
            cfg["multiscalefeature_outchannel"],
            downsampler2,
        )
        self.ffusion = FusionNet_upscale(cfg["multiscalefeature_outchannel"][-4:])
        self.heatmap = Heatmap(cfg["multiscalefeature_outchannel"][-4:-1])

    def forward(self, img):
        x = self.resconv(img)
        print(x)
        outputs_f = self.multiscalef(x)
        xf1 = self.ffusion(outputs_f[-4:])  # (B, 32, 256, 256)
        target = self.heatmap(xf1)
        return target


def HeatMaptoImg(heatmap, img, thre=0.5, img_dislocate_size=4):
    """
    optimize the heatmap to decrese the overlap area of different scale of heatmap.
    Args:
        heatmap:float32
            (2B, 1, 8, 8)
            (2B, 1, 16, 16)
            (2B, 1, 32, 32)
        img:(B, 1, H, W)
        thre: threshold to activate heatmap
        img_dislocate_size: the scale of dislocate of the second part of the batch.
    Returns:
        cutted_imgs:
            [scale:(32,16,8)[B batchs[(M, 32, 32)or(N, 16, 16)or(X, 8, 8)]]]
        bboxs:
            [scale:[batchs:M tensor(y1, x1, y2, x2)]](32, 16, 8)
    """
    # # sift area
    # sumer = sum_Conv(2, 2, 0)
    # sumer = sumer.to(img.device)
    # for i in range(len(heatmap) - 1):
    #     heatmap_ = heatmap[i + 1] > thre
    #     heatmap_ = heatmap_.type(torch.float32)
    #     heatmap[i] = heatmap[i] > thre
    #     heatmap_ = sumer(heatmap_)
    #     heatmap[i] = heatmap[i] & torch.where(
    #         heatmap_ == 1, torch.zeros_like(heatmap[i], dtype=torch.bool), torch.ones_like(heatmap[i], dtype=torch.bool)
    #     )
    #     heatmap_ = heatmap_.repeat(1, 1, 2, 2)
    #     heatmap[i + 1] = heatmap[i + 1] > thre
    #     heatmap[i + 1] = heatmap[i + 1] & torch.where(
    #         heatmap_ >= 1,
    #         torch.zeros_like(heatmap[i + 1], dtype=torch.bool),
    #         torch.ones_like(heatmap[i + 1], dtype=torch.bool),
    #     )

    # make heatmap bool
    for i in range(len(heatmap)):
        heatmap[i] = heatmap[i] > thre

    # bbox [scale:[batchs:tensor(x1, y1, x2, y2)]]
    B, _, _, _ = heatmap[0].shape
    B = int(B/ 2)
    bboxs = [[[] for i in range(B)] for j in range(len(heatmap))]
    scale = 8
    for i in range(len(heatmap) - 1, -1, -1):
        batch_indices, _, row_indices, col_indices = torch.where(heatmap[i] == 1)
        for idx in range(batch_indices.shape[0]):
            coor = (
                torch.tensor([row_indices[idx], col_indices[idx], row_indices[idx] + 1, col_indices[idx] + 1]) * scale
            )
            # coor[:2] -= int(scale*0.25)
            # coor[2:] += int(scale*0.25)
            if batch_indices[idx] >= B:
                coor = coor - img_dislocate_size
                min_coor, min_idx = torch.min(coor, dim=0)
                while min_coor < 0:
                    coor[min_idx] += 0 - min_coor
                    coor[min_idx+2] += 0 - min_coor 
                    min_coor, min_idx = torch.min(coor, dim=0)
                
            bboxs[i][batch_indices[idx] % B].append(coor)

        scale = scale * 2

    # cutting the target area [scale:[batch:, M*(H*W)]]
    multiscale = [[[] for i in range(B)] for j in range(len(heatmap))]
    for i in range(len(bboxs)):
        for j in range(B):
            for z in bboxs[i][j]:
                target_area = img[j, 0, z[0] : z[2], z[1] : z[3]].unsqueeze(0).unsqueeze(0)
                # local brightness normal
                min_val,_ = torch.min(target_area.reshape(-1), dim=0)
                max_val,_ = torch.max(target_area.reshape(-1), dim=0)
                target_area = (target_area-min_val.view(-1,1,1,1))/(max_val-min_val+1e-5).view(-1,1,1,1)
                multiscale[i][j].append(target_area)
            multiscale[i][j] = torch.concatenate(multiscale[i][j], 0) if len(multiscale[i][j]) > 0 else torch.tensor([], device=img.device)

    return multiscale, bboxs


def NoisePadding(img, noise_bg, x_coords, y_coords):
    """
    Args:
        img:(M, 1, size, size)
        noise_bg:(1, 32, 32)
        x_coords:(M,)
        y_coords:(M,)
    Returns:
        padded_img:
            all croped_img transformed to 32*32.(M, 1, 32, 32)
    """
    M, _, S, _ = img.shape
    LS = noise_bg.shape[-1]

    padded_img = torch.zeros((M, 1, LS, LS), device=img.device)
    padded_img[:] = noise_bg
    # 创建索引张量来选择正确的区域
    
    batch_indices = torch.arange(M).view(M, 1, 1, 1)  # 用于沿批处理维度进行索引
    x_indices = x_coords.view(M, 1, 1, 1).expand(M, 1, S, S)
    y_indices = y_coords.view(M, 1, 1, 1).expand(M, 1, S, S)

    # 创建一个索引网格，用于将子图嵌入到大图中
    grid_x, grid_y = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')
    grid_x = grid_x.expand(M, 1, S, S)
    grid_y = grid_y.expand(M, 1, S, S)

    # 使用高级索引来嵌入子图
    padded_img[batch_indices, 0, x_indices + grid_x, y_indices + grid_y] = img

    return padded_img

def RecoveryImgByCoor(img, x_coords, y_coords, size):
    if size == 32:
        return img
    if img.shape[0] == 0:
        return img
    M, S = img.shape[0], int(size)

    # res = torch.zeros((M, 1, S, S))
    # for i in range(M):
    #     print(x_coords[i], y_coords[i])
    #     res[i,0] = img[i, 0, x_coords[i]:x_coords[i]+S, y_coords[i]:y_coords[i]+S]
    # return res
    batch_indices = torch.arange(M).view(M, 1, 1, 1)  # 用于沿批处理维度进行索引
    x_indices = x_coords.view(M, 1, 1, 1).expand(M, 1, S, S)
    y_indices = y_coords.view(M, 1, 1, 1).expand(M, 1, S, S)

    # 创建一个索引网格，用于将子图嵌入到大图中
    grid_x, grid_y = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')
    grid_x = grid_x.expand(M, 1, S, S)
    grid_y = grid_y.expand(M, 1, S, S)

    # 使用高级索引来嵌入子图
    return img[batch_indices, 0, x_indices + grid_x, y_indices + grid_y]

def UnifyShape(img, noise, to_size=32):
    """
    Args:
        img:
            (B, M, 1, 8, 8)
            (B, N, 1, 16, 16)
            (B, Y, 1, 32, 32)
        noise:
            (B, 1, 32, 32)
        to_size: 
            target_size
    Returns:
        img:
            all croped_img transformed to 32*32.(num, 1, 32, 32)
        shape_idx:
            original shape of img, to make sure we reconstrcut the same consitution of img.
    """
    S, B = len(img), len(img[0])
    shape_idx = [[] for i in range(S)]
    img_tensor = torch.zeros((1, 1, to_size, to_size), device="cuda")
    for i in range(S):
        for j in range(B):
            if img[i][j].shape[0] == 0:
                shape_idx[i].append({"num": 0, "row_idx": 0, "col_idx": 0})
                continue
            num, _, size, _ = img[i][j].shape
            if size < to_size:
                coords = torch.randint(0,to_size - size,(num,2))
                bg = NoisePadding(img[i][j], noise[j], coords[:,0], coords[:,1])
                img_tensor = torch.concatenate((img_tensor, bg), dim=0)
                shape_idx[i].append({"num": num, "row_idx": coords[:,0], "col_idx": coords[:,1]})
            else:
                shape_idx[i].append({"num": num, "row_idx": 0, "col_idx": 0})
                img_tensor = torch.concatenate((img_tensor, img[i][j]), dim=0)
    return img_tensor[1:], shape_idx


def UnitLabels(multiscale_labels, bboxs, label_size):
    """
    reunit pieces of labels into one.
    Args:
        multiscale_labels:bool
            (B, M, 1, 8, 8)
            (B, N, 1, 16, 16)
            (B, Y, 1, 32, 32)
        bboxs:
            [scale:[batchs:M tensor(y1, x1, y2, x2)]]
        label_size: int
    Returns:
        label:
            (B,1,label_size,label_size)
    """
    B = len(multiscale_labels[0])
    label = torch.zeros(B, 1, label_size, label_size, device="cuda")
    cnt = torch.zeros(B, 1, label_size, label_size, device="cuda")
    for i in range(len(multiscale_labels) - 1, -1, -1):
        for j in range(B):
            for z in range(len(multiscale_labels[i][j])):
                label[j, 0, bboxs[i][j][z][0] : bboxs[i][j][z][2], bboxs[i][j][z][1] : bboxs[i][j][z][3]] += multiscale_labels[i][j][z, 0]
                cnt[j, 0, bboxs[i][j][z][0] : bboxs[i][j][z][2], bboxs[i][j][z][1] : bboxs[i][j][z][3]] += torch.ones_like(multiscale_labels[i][j][z, 0], device="cuda")
    label = label / (cnt+1e-10)
    return label


class sum_Conv(nn.Module):
    """
    sum all value of the region
    inputs:
        x:(B, C, H, W)
    """

    def __init__(self, kernel, stride, padding):
        super(sum_Conv, self).__init__()
        self.sumer = nn.Conv2d(1, 1, kernel, stride, padding, bias=False)
        self.sumer.weight.data = torch.ones((1, 1, kernel, kernel))
        self.sumer.weight.requires_grad = False  # 不需要梯度更新

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)
        res = self.sumer(x)
        res = res.view(B, C, res.shape[-2], res.shape[-1])
        return res


class ConvT2d_Bn_Relu(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int = 3):
        super(ConvT2d_Bn_Relu, self).__init__()
        self.convT = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, 2, 1, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.convT(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Learning_Block1(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Learning_Block1, self).__init__()
        self.conv1 = Conv2d_Bn_Relu(in_channel, out_channel, kernel_size, 2, 1)  # (B,C,16,16)
        self.conv2 = Conv2d_Bn_Relu(out_channel, out_channel * 2, kernel_size, 2, 1)  # (B,C,8,8)
        self.conv3 = Conv2d_Bn_Relu(out_channel * 2, out_channel * 4, kernel_size, 2, 1)  # (B,C,4,4)
        # self.conv4 = Conv2d_Bn_Relu(out_channel * 4, out_channel * 8, kernel_size, 2, 1)  # (B,C,2,2)

        # self.convT4 = ConvT2d_Bn_Relu(out_channel * 8, out_channel * 4, kernel_size)  # (B,C,8,8)
        self.convT3 = ConvT2d_Bn_Relu(out_channel * 4, out_channel * 2, kernel_size)  # (B,C,8,8)
        self.convT2 = ConvT2d_Bn_Relu(out_channel * 2, out_channel, kernel_size)  # (B,C,16,16)
        self.convT1 = ConvT2d_Bn_Relu(out_channel, in_channel, kernel_size)  # (B,C,32,32)

    def forward(self, img):
        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # x4 = self.conv4(x3)
        # xT3 = self.convT4(x4)
        # x3 = torch.min(xT3, x3)
        xT2 = self.convT3(x3)
        x2 = torch.min(xT2, x2)
        xT1 = self.convT2(x2)
        x1 = torch.min(xT1, x1)
        xT0 = self.convT1(x1)
        res = torch.min(xT0, img)
        return res


class Learning_Block2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Learning_Block2, self).__init__()
        self.conv1 = Conv2d_Bn_Relu(in_channel, out_channel, kernel_size, 2, 1)  # (B,C,16,16)
        self.conv2 = Conv2d_Bn_Relu(out_channel, out_channel * 2, kernel_size, 2, 1)  # (B,C,8,8)
        self.conv3 = Conv2d_Bn_Relu(out_channel * 2, out_channel * 4, kernel_size, 2, 1)  # (B,C,4,4)
        self.conv4 = Conv2d_Bn_Relu(out_channel * 4, out_channel * 8, kernel_size, 2, 1)  # (B,C,2,2)

        self.convT4 = ConvT2d_Bn_Relu(out_channel * 8, out_channel * 4, kernel_size)  # (B,C,8,8)
        self.convT3 = ConvT2d_Bn_Relu(out_channel * 4, out_channel * 2, kernel_size)  # (B,C,8,8)
        self.convT2 = ConvT2d_Bn_Relu(out_channel * 2, out_channel, kernel_size)  # (B,C,16,16)
        self.convT1 = ConvT2d_Bn_Relu(out_channel, in_channel, kernel_size)  # (B,C,32,32)

    def forward(self, img):
        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        xT3 = self.convT4(x4)
        x3 = torch.min(xT3, x3)
        xT2 = self.convT3(x3)
        x2 = torch.min(xT2, x2)
        xT1 = self.convT2(x2)
        x1 = torch.min(xT1, x1)
        xT0 = self.convT1(x1)
        res = torch.min(xT0, img)
        return res


class Learning_Block3(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Learning_Block3, self).__init__()
        self.conv1 = Conv2d_Bn_Relu(in_channel, out_channel, kernel_size, 2, 1)  # (B,C,16,16)
        self.conv2 = Conv2d_Bn_Relu(out_channel, out_channel * 2, kernel_size, 2, 1)  # (B,C,8,8)
        self.conv3 = Conv2d_Bn_Relu(out_channel * 2, out_channel * 3, kernel_size, 2, 1)  # (B,C,4,4)
        self.conv4 = Conv2d_Bn_Relu(out_channel * 3, out_channel * 4, kernel_size, 2, 1)  # (B,C,2,2)

        self.convT4 = ConvT2d_Bn_Relu(out_channel * 4, out_channel * 3, kernel_size)  # (B,C,8,8)
        self.convT3 = ConvT2d_Bn_Relu(out_channel * 3, out_channel * 2, kernel_size)  # (B,C,8,8)
        self.convT2 = ConvT2d_Bn_Relu(out_channel * 2, out_channel, kernel_size)  # (B,C,16,16)
        self.convT1 = ConvT2d_Bn_Relu(out_channel, in_channel, kernel_size)  # (B,C,32,32)

    def forward(self, img):
        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        xT3 = self.convT4(x4)
        x3 = torch.min(xT3, x3)
        xT2 = self.convT3(x3)
        x2 = torch.min(xT2, x2)
        xT1 = self.convT2(x2)
        x1 = torch.min(xT1, x1)
        xT0 = self.convT1(x1)
        res = torch.min(xT0, img)
        return res


class LearningConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(LearningConv, self).__init__()
        self.linear1 = Conv2d_Bn_Relu(in_channel, 8, kernel_size, 1, padding=1)
        self.block2 = Learning_Block1(8, 16, kernel_size)
        self.block3 = Learning_Block1(8, 16, kernel_size)
        self.block4 = Learning_Block1(8, 16, kernel_size)
        self.linear6 = Conv2d_Bn_Relu(8, out_channel, kernel_size, 1, 1)

    def forward(self, img):
        x = self.linear1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.linear6(x)
        return x


class LearningConv_longer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(LearningConv_longer, self).__init__()
        self.linear1 = Conv2d_Bn_Relu(in_channel, 4, kernel_size, 1, padding=1)
        self.block2 = Learning_Block1(4, 8, kernel_size)
        self.block3 = Learning_Block1(4, 8, kernel_size)
        self.block4 = Learning_Block1(4, 8, kernel_size)
        self.block5 = Learning_Block1(4, 8, kernel_size)
        self.linear6 = Conv2d_Bn_Relu(4, out_channel, kernel_size, 1, 1)

    def forward(self, img):
        x = self.linear1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.linear6(x)
        return x


class LearningConv_deeper(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(LearningConv_deeper, self).__init__()
        self.linear1 = Conv2d_Bn_Relu(in_channel, 4, kernel_size, 1, padding=1)
        self.block2 = Learning_Block1(4, 8, kernel_size)
        self.block3 = Learning_Block1(4, 8, kernel_size)
        self.block4 = Learning_Block1(4, 8, kernel_size)
        self.linear6 = Conv2d_Bn_Relu(4, out_channel, kernel_size, 1, 1)

    def forward(self, img):
        x = self.linear1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.linear6(x)
        return x


class LearningConv_shorter(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(LearningConv_shorter, self).__init__()
        self.linear1 = Conv2d_Bn_Relu(in_channel, 8, kernel_size, 1, padding=1)
        self.block2 = Learning_Block1(8, 16, kernel_size)
        self.block3 = Learning_Block1(8, 16, kernel_size)
        # self.block4 = Learning_Block1(4, 8, kernel_size)
        self.linear6 = Conv2d_Bn_Relu(8, out_channel, kernel_size, 1, 1)

    def forward(self, img):
        x = self.linear1(img)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        x = self.linear6(x)
        return x


class LearningConv_thinner(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(LearningConv_thinner, self).__init__()
        self.linear1 = Conv2d_Bn_Relu(in_channel, 8, kernel_size, 1, padding=1)
        self.block2 = Learning_Block2(8, 8, kernel_size)
        self.block3 = Learning_Block2(8, 8, kernel_size)
        self.block4 = Learning_Block2(8, 8, kernel_size)
        self.linear6 = Conv2d_Bn_Relu(8, out_channel, kernel_size, 1, 1)

    def forward(self, img):
        x = self.linear1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.linear6(x)
        return x
    

class LearningConv_thinner2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(LearningConv_thinner2, self).__init__()
        self.linear1 = Conv2d_Bn_Relu(in_channel, 8, kernel_size, 1, padding=1)
        self.block2 = Learning_Block3(8, 16, kernel_size)
        self.block3 = Learning_Block3(8, 16, kernel_size)
        self.block4 = Learning_Block3(8, 16, kernel_size)
        self.linear6 = Conv2d_Bn_Relu(8, out_channel, kernel_size, 1, 1)

    def forward(self, img):
        x = self.linear1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.linear6(x)
        return x


class LearningConv_stronger(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(LearningConv_stronger, self).__init__()
        self.linear1 = Conv2d_Bn_Relu(in_channel, 8, kernel_size, 1, padding=1)
        self.block2 = Learning_Block2(8, 16, kernel_size)
        self.block3 = Learning_Block2(8, 16, kernel_size)
        self.block4 = Learning_Block2(8, 16, kernel_size)
        self.linear6 = Conv2d_Bn_Relu(8, out_channel, kernel_size, 1, 1)

    def forward(self, img):
        x = self.linear1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.linear6(x)
        return x


class LocalSegment(nn.Module):
    def __init__(self, cfg=None):
        super(LocalSegment, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        self.resconv = LearningConv_thinner2(1, cfg["learning_conv_outchannel"])

        self.linear = DetectNet1(cfg["learning_conv_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)
        res = self.linear(x)
        return res


class ConvBlockDS(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlockDS, self).__init__()

        self.fe = Conv2d_Bn_Relu(in_channel, out_channel, 3, 1, 1)
        self.downsampler = Conv2d_Bn_Relu(out_channel, out_channel, 3, 2, 1)

    def forward(self, img):
        x = self.fe(img)
        res = self.downsampler(x)
        return res
    
class ConvBlockUS(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlockUS, self).__init__()

        self.fe = Conv2d_Bn_Relu(in_channel, out_channel, 3, 1, 1)
        self.downsampler = ConvT2d_Bn_Relu(out_channel, out_channel, 3)

    def forward(self, img):
        x = self.fe(img)
        res = self.downsampler(x)
        return res


class LocalSegBlock(nn.Module):
    def __init__(self, channel_list=[], cfg=None):
        super(LocalSegBlock, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")
        if len(channel_list) < 2:
            raise ValueError("parameter 'cfg' is not proper")

        encoder_list = []
        for i in range(1, len(channel_list)):
            encoder_list.append(ConvBlockDS(channel_list[i-1], channel_list[i]))
        self.encoder = nn.ModuleList(encoder_list)

        decoder_list = []
        for i in range(len(channel_list)-1, 1, -1):
            decoder_list.append(ConvBlockUS(channel_list[i], channel_list[i-1]))
        decoder_list.append(ConvBlockUS(channel_list[1], cfg["learning_conv_outchannel"]))
        self.decoder = nn.ModuleList(decoder_list)

        self.linear = DetectNet1(cfg["learning_conv_outchannel"], 1)

    def forward(self, img):
        data = [img,]
        for i in self.encoder:
            data.append(i(data[-1]))
        
        for i in range(len(self.decoder)-1):
            newf = self.decoder[i](data[-1])
            newf = torch.min(newf, data[len(self.encoder)-1-i])
            data.append(newf)
        data.append(self.decoder[len(self.decoder)-1](data[-1]))
        
        res = self.linear(data[-1])
        return res


class LocalSegment2(nn.Module):
    def __init__(self, cfg=None):
        super(LocalSegment2, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        self.fe1 = ConvBlockDS(1, 8)    #(16)
        self.fe2 = ConvBlockDS(8, 16)   #(8)
        self.fe3 = ConvBlockDS(16, 32)  #(4)
        self.fe4 = ConvBlockDS(32, 64)  #(2)
        # self.fe5 = ConvBlockDS(64, 128)  #(1)

        # self.feT5 = ConvBlockUS(128, 64) #(2)
        self.feT4 = ConvBlockUS(64, 32) #(4)
        self.feT3 = ConvBlockUS(32, 16) #(8)
        self.feT2 = ConvBlockUS(16, 8)  #(16)
        self.feT1 = ConvBlockUS(8, 8)   #(32)

        self.linear = DetectNet1(8, 1)

    def forward(self, img):
        x1 = self.fe1(img)
        x2 = self.fe2(x1)
        x3 = self.fe3(x2)
        x4 = self.fe4(x3)
        # x5 = self.fe5(x4)
        # x4 = torch.min(self.feT5(x5), x4)
        x3 = torch.min(self.feT4(x4), x3)
        x2 = torch.min(self.feT3(x3), x2)
        x1 = torch.min(self.feT2(x2), x1)
        x = self.feT1(x1)
        
        res = self.linear(x)
        return res


class LocalSegment3(nn.Module):
    """
    Args:
        inputs:
            (B,1,8,8)
            (B,1,16,16)
            (B,1,32,32)
        outputs:
            (B,1,8,8)
            (B,1,16,16)
            (B,1,32,32)
    """
    def __init__(self, cfg=None):
        super(LocalSegment3, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")
        
        self.localseg8 = LocalSegBlock([1,8,16,32], cfg=cfg)

        self.localseg16 = LocalSegBlock([1,8,16,32,64], cfg=cfg)

        self.localseg32 = LocalSegBlock([1,8,16,32,64,128], cfg=cfg)

    def forward(self, multiscale_img):
        res8 = self.localseg8(multiscale_img[0])
        res16 = self.localseg16(multiscale_img[1])
        res32 = self.localseg32(multiscale_img[2])
        return res8, res16, res32


class TwoTaskNetWithLoss(nn.Module):
    def __init__(self, cfg=None):
        super(TwoTaskNetWithLoss, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")
 
        self.cfg = cfg

        self.net_heatmap = Heatmap_net2(cfg=cfg)
        self.net_localseg = LocalSegment3(cfg=cfg)

        # self.heatmap_mse = Heatmap_MSE(cfg)
        self.heatmap_softiou = Heatmap_SoftIoU(cfg)
        self.softiou = SoftLoULoss()

    def forward(self, img, label, noise):
        img, label = self.dislocate(4, img, label)

        target = self.net_heatmap(img)

        label = (label > self.cfg["label_vague_threshold"]).type(torch.float32)
        heatmap_loss = self.heatmap_softiou(target, label)

        multiscale_imgs, bboxs = HeatMaptoImg(target, img, self.cfg["heatmap_activation_threshold"])
        # imgs, shape_idx = UnifyShape(multiscale_imgs, noise, self.cfg["target_detection_size"],)
        
        # record the num of each pict in order to recovery pitch prediction to full prediction.
        num = [[] for i in range(len(multiscale_imgs))]
        for i in range(len(multiscale_imgs)):
            for j in range(len(multiscale_imgs[i])):
                num[i].append(multiscale_imgs[i][j].shape[0])
            multiscale_imgs[i] = torch.concatenate(multiscale_imgs[i], dim=0)          

        pred = self.net_localseg(multiscale_imgs[::-1])

        # reconstruct the list strcuture
        results = [[[] for j in range(len(num[i]))] for i in range(len(num))]
        for i in range(len(num)):
            cnt = 0
            for j in range(len(num[i])):
                cnt_new = num[i][j] + cnt
                results[i][j] = pred[2-i][cnt:cnt_new]
                cnt = cnt_new

        pred = UnitLabels(results, bboxs, 256)

        softiou_loss = self.softiou(pred, label[:int(label.shape[0]/2)])

        # # 显示图片
        # for i in range(len(multiscale_imgs)):
        #     for j in range(len(multiscale_imgs[0])):
        #         print(target[i][j])
        #         # for z in range(results[i][j].shape[0]):
        #         ori_pict = np.array(img.cpu())
        #         # target_pict = np.array(multiscale_imgs[i][j][z].cpu())
        #         # target_pict = np.array(imgs[cnt].cpu())
        #         # pred_pict = np.array(pred.detach().cpu())
        #         pred_pict = np.array(pred[j].detach().cpu())
        #         label_ = np.array(label[j].cpu())

        #         fig, axes = plt.subplots(3)
        #         axes[0].imshow(ori_pict[j, 0], cmap="gray")
        #         axes[0].axis("off")  # 关闭坐标轴显示

        #         # axes[1].imshow(target_pict, cmap="gray")
        #         # axes[1].axis("off")  # 同样关闭坐标轴显示

        #         axes[1].imshow(pred_pict[0], cmap="gray")
        #         axes[1].axis("off")  # 同样关闭坐标轴显示

        #         axes[2].imshow(label_[0], cmap="gray")
        #         axes[2].axis("off")  # 同样关闭坐标轴显示

        #         plt.tight_layout()
        #         plt.show()
        #         a = input()

        return heatmap_loss, softiou_loss, pred

    def dislocate(self, size, img, label):
        """
        Args:
            size: the scale of how much we should dislocate comparing with original picture.
            img: (B, 1, S, S)
            label: (B, 1, S, S)
        Returns:
            imgs: (B*2, 1, S, S)
        """
        img2 = torch.zeros_like(img)
        img2[..., size:, size:] = img[..., :-size, :-size]
        label2 = torch.zeros_like(label)
        label2[..., size:, size:] = label[..., :-size, :-size]
        return torch.concatenate((img, img2), dim=0), torch.concatenate((label, label2), dim=0)
