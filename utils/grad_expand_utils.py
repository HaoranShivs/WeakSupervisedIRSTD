import torch
import torch.nn.functional as F

import numpy as np


def get_sobel_filter_33():
    op_0 = torch.tensor([[[[0.125, 1.0, 0.125], 
                           [0.0, 0.0, 0.0], 
                           [-0.125, -1.0, -0.125]]]])
    op_90 = torch.tensor([[[[-0.125, 0.0, 0.125], 
                            [-1.0, 0.0, 1.0], 
                            [-0.125, 0.0, 0.125]]]])
    op_45 = torch.tensor([[[[0.0, 0.125, 1.0], 
                            [-0.125, 0.0, 0.125], 
                            [-1.0, -0.125, 0.0]]]])
    op_135 = torch.flip(op_45, dims=[2])
    op_180 = torch.flip(op_0, dims=[2, 3])
    op_225 = torch.flip(op_45, dims=[2, 3])
    op_270 = torch.flip(op_90, dims=[2, 3])
    op_315 = torch.flip(op_45, dims=[3])
    x1_ = np.cos(15 * np.pi / 180)
    x2_ = np.cos(30 * np.pi / 180)
    x1 = x1_ / np.sqrt(x1_**2 + x2_**2)
    x2 = x2_ / np.sqrt(x1_**2 + x2_**2)
    op_15 = op_0 * x1 + op_45 * x2
    op_30 = op_0 * x2 + op_45 * x1
    op_60 = op_45 * x1 + op_90 * x2
    op_75 = op_45 * x2 + op_90 * x1
    op_105 = op_90 * x1 + op_135 * x2
    op_120 = op_90 * x2 + op_135 * x1
    op_150 = op_135 * x1 + op_180 * x2
    op_165 = op_135 * x2 + op_180 * x1
    op_195 = op_180 * x1 + op_225 * x2
    op_210 = op_180 * x2 + op_225 * x1
    op_240 = op_225 * x1 + op_270 * x2
    op_255 = op_225 * x2 + op_270 * x1
    op_285 = op_270 * x1 + op_315 * x2
    op_300 = op_270 * x2 + op_315 * x1
    op_330 = op_315 * x1 + op_0 * x2
    op_345 = op_315 * x2 + op_0 * x1
    sobel_33 = torch.cat(
        [
            op_0, op_15, op_30, op_45, op_60, op_75, op_90, op_105,
            op_120, op_135, op_150, op_165, op_180, op_195, op_210,
            op_225, op_240, op_255, op_270, op_285, op_300, op_315,
            op_330, op_345,
        ],
        dim=0,
    )
    return sobel_33 / 1.25

def get_sobel_filter_22():
    op_0 = torch.tensor([[[[0, 1.0], [0.0, -1.0]]]])
    op_90 = torch.tensor([[[[0.0, 0.0], [-1.0, 1.0]]]])
    op_45 = torch.tensor([[[[0.0, 1.0], [-1.0, 0.0]]]])
    op_135 = torch.flip(op_45, dims=[2])
    op_180 = torch.flip(op_0, dims=[2, 3])
    op_225 = torch.flip(op_45, dims=[2, 3])
    op_270 = torch.flip(op_90, dims=[2, 3])
    op_315 = torch.flip(op_45, dims=[3])
    x1_ = np.cos(15 * np.pi / 180)
    x2_ = np.cos(30 * np.pi / 180)
    x1 = x1_ / np.sqrt(x1_**2 + x2_**2)
    x2 = x2_ / np.sqrt(x1_**2 + x2_**2)
    op_15 = op_0 * x1 + op_45 * x2
    op_30 = op_0 * x2 + op_45 * x1
    op_60 = op_45 * x1 + op_90 * x2
    op_75 = op_45 * x2 + op_90 * x1
    op_105 = op_90 * x1 + op_135 * x2
    op_120 = op_90 * x2 + op_135 * x1
    op_150 = op_135 * x1 + op_180 * x2
    op_165 = op_135 * x2 + op_180 * x1
    op_195 = op_180 * x1 + op_225 * x2
    op_210 = op_180 * x2 + op_225 * x1
    op_240 = op_225 * x1 + op_270 * x2
    op_255 = op_225 * x2 + op_270 * x1
    op_285 = op_270 * x1 + op_315 * x2
    op_300 = op_270 * x2 + op_315 * x1
    op_330 = op_315 * x1 + op_0 * x2
    op_345 = op_315 * x2 + op_0 * x1
    sobel_22 = torch.cat(
        [
            op_0, op_15, op_30, op_45, op_60, op_75, op_90, op_105,
            op_120, op_135, op_150, op_165, op_180, op_195, op_210,
            op_225, op_240, op_255, op_270, op_285, op_300, op_315,
            op_330, op_345,
        ],
        dim=0,
    )
    return sobel_22


__sobel_33 = get_sobel_filter_33()
__sobel_22 = get_sobel_filter_22()


def img_gradient3(image_batch):
    dilation = 1
    img_padded = F.pad(image_batch, (dilation * 2, dilation * 2, dilation * 2, dilation * 2), mode="replicate")
    grad = F.conv2d(img_padded, __sobel_33, dilation=dilation)  # (B, 24, S+2, S+2)
    # 错位处理
    _, _, H, W = image_batch.shape
    grad_ = []
    for i in range(24):
        angle = torch.tensor(
            [
                90 - i * 15,
            ],
            device=image_batch.device,
        )
        x = torch.round(torch.cos_(angle * np.pi / 180) * dilation).type(torch.int64)
        y = torch.round(torch.sin_(angle * np.pi / 180) * dilation).type(torch.int64)
        grad_.append(grad[:, i, dilation + y : dilation + y + H, dilation - x : dilation - x + W])
    grad = torch.stack(grad_, dim=1)
    zeros_tensor = torch.zeros_like(grad)
    grad = torch.where(grad > 0, grad, zeros_tensor)

    return grad

def img_gradient2(image_batch):
    B, _, H, W = image_batch.shape
    img_padded = F.pad(image_batch, (1, 1, 1, 1), mode="replicate")
    grad = F.conv2d(img_padded, __sobel_22)  # (B, 24, H+1, W+1)
    # 修正梯度对应像素的位置
    grad = torch.cat(
        [
            grad[:, 0:5, 1 : H + 1, :W],
            grad[:, 5:11, :H, :W],
            grad[:, 11:17, :H, 1 : W + 1],
            grad[:, 17:23, 1 : H + 1, 1 : W + 1],
            grad[:, 23:, 1 : H + 1, :W],
        ],
        dim=1,
    )

    zeros_tensor = torch.zeros_like(grad)
    grad = torch.where(grad > 0, grad, zeros_tensor)

    return grad

def img_gradient5(image_batch):
    dilation = 2
    img_padded = F.pad(image_batch, (dilation * 2, dilation * 2, dilation * 2, dilation * 2), mode="replicate")
    grad = F.conv2d(img_padded, __sobel_33, dilation=dilation)  # (B, 24, S+6, S+6)
    # 错位处理
    _, _, H, W = image_batch.shape
    grad_ = []
    for i in range(24):
        angle = torch.tensor(
            [
                90 - i * 15,
            ],
            device=image_batch.device,
        )
        x = torch.round(torch.cos_(angle * np.pi / 180) * dilation).type(torch.int64)
        y = torch.round(torch.sin_(angle * np.pi / 180) * dilation).type(torch.int64)
        grad_.append(grad[:, i, dilation + y : dilation + y + H, dilation - x : dilation - x + W])
    grad = torch.stack(grad_, dim=1)
    zeros_tensor = torch.zeros_like(grad)
    grad = torch.where(grad > 0, grad, zeros_tensor)

    return grad

def local_max_gradient(tensor):
    """
    对形状为 (B, C, S, S) 的张量进行高斯滤波。

    参数:
    tensor -- 输入张量

    返回:
    filtered_tensor -- 滤波后的张量
    """
    _, _, H, W = tensor.shape
 
    # 使用分组卷积计算所有方向的梯度
    C = 24
    grads_forward = F.conv2d(tensor, __sobel_22, padding=1, groups=C)  # 形状为 (B, 24, H, W)
    grads_backward = F.conv2d(tensor, __sobel_22.flip(dims=[2, 3]), padding=1, groups=C)  # 反向梯度

    grads_forward = torch.cat(
        [
            grads_forward[:, 0:5, 1:, :W],
            grads_forward[:, 5:11, :H, :W],
            grads_forward[:, 11:17, :H, 1:],
            grads_forward[:, 17:23, 1:, 1:],
            grads_forward[:, 23:, 1:, :W],
        ],
        dim=1,
    )
    grads_backward = torch.cat(
        [
            grads_backward[:, 0:5, :H, 1:],
            grads_backward[:, 5:11, 1:, 1:],
            grads_backward[:, 11:17, 1:, :W],
            grads_backward[:, 17:23, :H, :W],
            grads_backward[:, 23:, :H, 1:],
        ],
        dim=1,
    )

    # 边界处理
    grads_forward[:, :, :1, :], grads_backward[:, :, :1, :] = 0, 0
    grads_forward[:, :, :, :1], grads_backward[:, :, :, :1] = 0, 0
    grads_forward[:, :, H - 1 :, :], grads_backward[:, :, H - 1 :, :] = 0, 0
    grads_forward[:, :, :, W - 1 :], grads_backward[:, :, :, W - 1 :] = 0, 0

    mask = ((grads_forward > 0) * (grads_backward > 0)).type(torch.float32)

    return mask

# def gis(grad, grad_mask):
#     op_0 = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]], dtype=grad.dtype, device=grad.device)
#     op_45 = torch.tensor([[[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]], dtype=grad.dtype, device=grad.device)
#     op_90 = torch.tensor([[[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]], dtype=grad.dtype, device=grad.device)
#     op_135 = torch.flip(op_45, dims=[2])
#     op_180 = torch.flip(op_0, dims=[2, 3])
#     op_225 = torch.flip(op_45, dims=[2, 3])
#     op_270 = torch.flip(op_90, dims=[2, 3])
#     op_315 = torch.flip(op_45, dims=[3])
#     op_15 = op_0 * 2 / 3 + op_45 * 1 / 3
#     op_30 = op_0 * 1 / 3 + op_45 * 2 / 3
#     op_60 = op_45 * 2 / 3 + op_90 * 1 / 3
#     op_75 = op_45 * 1 / 3 + op_90 * 2 / 3
#     op_105 = op_90 * 2 / 3 + op_135 * 1 / 3
#     op_120 = op_90 * 1 / 3 + op_135 * 2 / 3
#     op_150 = op_135 * 2 / 3 + op_180 * 1 / 3
#     op_165 = op_135 * 1 / 3 + op_180 * 2 / 3
#     op_195 = op_180 * 2 / 3 + op_225 * 1 / 3
#     op_210 = op_180 * 1 / 3 + op_225 * 2 / 3
#     op_240 = op_225 * 2 / 3 + op_270 * 1 / 3
#     op_255 = op_225 * 1 / 3 + op_270 * 2 / 3
#     op_285 = op_270 * 2 / 3 + op_315 * 1 / 3
#     op_300 = op_270 * 1 / 3 + op_315 * 2 / 3
#     op_330 = op_315 * 2 / 3 + op_0 * 1 / 3
#     op_345 = op_315 * 1 / 3 + op_0 * 2 / 3

#     # 将所有方向梯度算子堆叠成一个大卷积核
#     all_ops = torch.cat(
#         [
#             op_0, op_15, op_30, op_45, op_60, op_75, op_90, op_105,
#             op_120, op_135, op_150, op_165, op_180, op_195, op_210,
#             op_225, op_240, op_255, op_270, op_285, op_300, op_315,
#             op_330, op_345,
#         ],
#         dim=0,
#     )  # 形状为 (24, 1, 2, 2)

#     C = 24
#     # 使用梯度积分强度（Gradient Integral Strength, GIS）方案
#     masked_tensor = grad * (1 - grad_mask)
#     gis = F.conv2d(masked_tensor, all_ops, padding=1, groups=C)
#     gis = gis * grad_mask + grad * grad_mask

#     return gis

def gradient_expand_one_step(gradient):
    # 创建基础卷积核
    op_0 = torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]], dtype=gradient.dtype, device=gradient.device
    )
    op_45 = torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]], dtype=gradient.dtype, device=gradient.device
    )
    op_90 = torch.tensor(
        [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=gradient.dtype, device=gradient.device
    )
    op_135 = torch.flip(op_45, dims=[2])
    op_180 = torch.flip(op_0, dims=[2, 3])
    op_225 = torch.flip(op_45, dims=[2, 3])
    op_270 = torch.flip(op_90, dims=[2, 3])
    op_315 = torch.flip(op_45, dims=[3])

    x1_ = np.cos(15 * np.pi / 180)
    x2_ = np.cos(30 * np.pi / 180)
    x1 = x1_ / (x1_ + x2_)
    x2 = x2_ / (x1_ + x2_)
    op_15 = op_0 * x1 + op_45 * x2
    op_30 = op_0 * x2 + op_45 * x1
    op_60 = op_45 * x1 + op_90 * x2
    op_75 = op_45 * x2 + op_90 * x1
    op_105 = op_90 * x1 + op_135 * x2
    op_120 = op_90 * x2 + op_135 * x1
    op_150 = op_135 * x1 + op_180 * x2
    op_165 = op_135 * x2 + op_180 * x1
    op_195 = op_180 * x1 + op_225 * x2
    op_210 = op_180 * x2 + op_225 * x1
    op_240 = op_225 * x1 + op_270 * x2
    op_255 = op_225 * x2 + op_270 * x1
    op_285 = op_270 * x1 + op_315 * x2
    op_300 = op_270 * x2 + op_315 * x1
    op_330 = op_315 * x1 + op_0 * x2
    op_345 = op_315 * x2 + op_0 * x1
    
    # 将所有方向梯度算子堆叠成一个大卷积核
    all_ops = torch.cat(
        [
            op_0, op_15, op_30, op_45, op_60, op_75, op_90, op_105,
            op_120, op_135, op_150, op_165, op_180, op_195, op_210,
            op_225, op_240, op_255, op_270, op_285, op_300, op_315,
            op_330, op_345,
        ],
        dim=0,
    )  # 形状为 (24, 1, 2, 2)
    
    # 执行单次分组卷积
    gradient_ = F.conv2d(gradient, weight=all_ops, padding=1, groups=24)  # 每个输入通道独立处理

    return gradient_

def boundary4gradient_expand(gradient, zoom_rate=1e9):
    _, C, _, _ = gradient.shape

    op_0 = torch.tensor(
        [[[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=gradient.dtype, device=gradient.device
    )
    op_45 = torch.tensor(
        [[[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=gradient.dtype, device=gradient.device
    )
    op_90 = torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]], dtype=gradient.dtype, device=gradient.device
    )
    op_135 = torch.flip(op_45, dims=[2])
    op_180 = torch.flip(op_0, dims=[2, 3])
    op_225 = torch.flip(op_45, dims=[2, 3])
    op_270 = torch.flip(op_90, dims=[2, 3])
    op_315 = torch.flip(op_45, dims=[3])
    x1_ = np.cos(15 * np.pi / 180)
    x2_ = np.cos(30 * np.pi / 180)
    x1 = x1_ / np.sqrt(x1_**2 + x2_**2)
    x2 = x2_ / np.sqrt(x1_**2 + x2_**2)
    op_15 = op_0 * x1 + op_45 * x2
    op_30 = op_0 * x2 + op_45 * x1
    op_60 = op_45 * x1 + op_90 * x2
    op_75 = op_45 * x2 + op_90 * x1
    op_105 = op_90 * x1 + op_135 * x2
    op_120 = op_90 * x2 + op_135 * x1
    op_150 = op_135 * x1 + op_180 * x2
    op_165 = op_135 * x2 + op_180 * x1
    op_195 = op_180 * x1 + op_225 * x2
    op_210 = op_180 * x2 + op_225 * x1
    op_240 = op_225 * x1 + op_270 * x2
    op_255 = op_225 * x2 + op_270 * x1
    op_285 = op_270 * x1 + op_315 * x2
    op_300 = op_270 * x2 + op_315 * x1
    op_330 = op_315 * x1 + op_0 * x2
    op_345 = op_315 * x2 + op_0 * x1

    # 将所有方向梯度算子堆叠成一个大卷积核
    all_ops = torch.cat(
        [
            op_0, op_15, op_30, op_45, op_60, op_75, op_90, op_105,
            op_120, op_135, op_150, op_165, op_180, op_195, op_210,
            op_225, op_240, op_255, op_270, op_285, op_300, op_315,
            op_330, op_345,
        ],
        dim=0,
    )  # 形状为 (24, 1, 2, 2)

    # 执行单次分组卷积
    gradient_ = F.conv2d(gradient, weight=all_ops, padding=1, groups=24)  # 每个输入通道独立处理

    # 梯度比较
    gradient_ = torch.where(gradient > gradient_, gradient, gradient_) * (gradient_ > 0.)

    # 形成扩张终点
    gradient_2 = torch.zeros_like(gradient_)
    for i in range(C):
        t_idx = (i + C // 2) % C
        gradient_2[:, i] = gradient_[:, t_idx] * (-zoom_rate)
    return gradient_2

def sigmoid_mapping(tensor, alpha1, alpha2, alpha3=0.5):
    y = (1+alpha2)/(1 + torch.exp(-alpha1*(tensor-alpha3))) - alpha2/2
    return torch.clamp(y, min=0., max=1.)

def sigmoid_mapping2(tensor, alpha1, alpha2, alpha3=0.5):
    y = alpha2/(1 + torch.exp(-alpha1*(tensor-alpha3))) + (1-alpha2)
    return y

def sigmoid_mapping3(tensor, alpha1, alpha3=0.25):
    constant = torch.exp(torch.tensor(alpha1)*torch.tensor(alpha3))
    y = (1/(1 + torch.exp(-alpha1*(tensor-alpha3))) - 1/(1 + constant)) * (1 + constant)/constant
    return y

def grad_multi_scale_fusion(tensor, weights):
    return tensor * weights + 1 - weights

