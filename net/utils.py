import torch
import math


def hide_channels(out_channel:int, num:int):
    res = []
    i = 2
    for j in range(num):
        hide_channel = int(out_channel/i)
        i = i * 2
        if hide_channel > 1:
            res.insert(0,hide_channel)
        else:
            res.append(out_channel)
    return res


def gaussian_kernel(channel, kernel_size, sigma):
    """创建一个高斯核"""
    # 创建一个一维高斯核
    kernel_1d = torch.Tensor([math.exp(-z ** 2.0 / (2 * sigma ** 2)) for z in range(-kernel_size // 2 + 1, kernel_size // 2 + 1)])
    # 将一维高斯核扩展为二维
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    # 归一化
    kernel_2d /= kernel_2d.sum()
    kernel_2d[kernel_size // 2, kernel_size // 2] = -kernel_2d[kernel_size // 2, kernel_size // 2]
    # 整个卷积核参数
    params = torch.zeros((channel, channel, kernel_size, kernel_size))
    for i in range(channel):
        params[i, i] = kernel_2d
    return params
