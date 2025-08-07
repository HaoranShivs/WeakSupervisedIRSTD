import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as Data

import os
import os.path as osp
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

from scipy.signal import find_peaks as fpk
from scipy.ndimage import label as ndlabel
from scipy import ndimage

# from models import get_model
from dataprocess.sirst import NUDTDataset, IRSTD1kDataset
from net.attentionnet import attenMultiplyUNet_withloss
from net.dnanet import DNANet_withloss, Res_CBAM_block
from pseudo_label_generate import img_gradient2, img_gradient3, img_gradient5, local_max_gradient, \
    gradient_expand_one_step, gradient_expand__one_step_boundary, boundary4gradient_expand, compute_histogram, \
        smooth_histogram, compute_histogram_slope, robust_min_max, apply_crf, object_closed_score, sigmoid_mapping3, \
        grad_multi_scale_fusion, hist_mapping, mapping_4_crf, mapping_4_crf_v3
from utils.sum_val_filter import sum_val_filter
from utils.utils import extract_local_windows, gaussian_blurring_2D, iou_score
from utils.refine import dilate_mask
# fuctions to produce pesudo label


def gaussian_kernel(size, sigma): 
    coords = torch.arange(size, dtype=torch.float32)
    coords -= (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def gradient_expand_one_step_boundary(gradient, boundary):
    gradient_ = gradient_expand_one_step(gradient)
    gradient_ = gradient_ + boundary
    # plt.figure(figsize=(15, 5))
    # plt.subplot(131), plt.imshow(torch.sum(gradient_[0], dim=0), cmap='gray')
    # plt.subplot(132), plt.imshow(torch.sum(gradient[0], dim=0), cmap='gray')
    # plt.subplot(133), plt.imshow(torch.sum(boundary[0], dim=0), cmap='gray') # torch.sum(img_gradient_3[0], dim=0)
    # plt.show()
    gradient_ = torch.where(gradient > gradient_, gradient, gradient_) * (gradient_ > -1e-4)
    return gradient_


def find_center_of_mask(mask):
    """
    找到 mask 中值为 1 的像素的中心索引。

    参数:
        mask (torch.Tensor): 形状为 [1, 1, H, W] 的张量，值为 0 或 1。

    返回:
        tuple: 中心像素的索引 (center_y, center_x)。
    """
    # 检查输入形状是否符合要求
    if mask.shape[0] != 1 or mask.shape[1] != 1:
        raise ValueError("Input mask shape must be [1, 1, H, W].")
    
    # 获取高度和宽度
    _, _, H, W = mask.shape

    # 将 mask 展平为二维 [H, W]
    mask = mask.squeeze(0).squeeze(0)  # 形状变为 [H, W]

    # 找到值为 1 的像素的索引
    y_indices, x_indices = torch.where(mask == 1)

    # 如果没有值为 1 的像素，返回 None
    if y_indices.numel() == 0:
        return None

    # 计算中心索引
    center_y = y_indices.float().mean().item()
    center_x = x_indices.float().mean().item()

    return torch.round(center_y), torch.round(center_x)


def sigmoid_mapping(tensor, alpha1, alpha2, alpha3=0.5):
    y = (1+alpha2)/(1 + torch.exp(-alpha1*(tensor-alpha3))) - alpha2/2
    return torch.clamp(y, min=0., max=1.)

def sigmoid_mapping2(tensor, alpha1, alpha2, alpha3=0.5):
    y = alpha2/(1 + torch.exp(-alpha1*(tensor-alpha3))) + (1-alpha2)
    return y


def gradient_expand_one_size(region):
    """
    """
    # 梯度形成
    img_gradient_1 = img_gradient2(region)  # 2*2 sober
    img_gradient_2 = img_gradient3(region)  # 3*3 sober
    img_gradient_3 = img_gradient5(region)  # 5*5 sobel
    # # 取局部最大值，将梯度的边缘最大化
    # img_gradient_ = img_gradient_1 + img_gradient_2
    # grad_mask = local_max_gradient(img_gradient_)

    # # 取值映射，突出中灰度的区分度
    # min_value1, max_value1 = robust_min_max(img_gradient_1, 0.001, 0.1)
    # min_value2, max_value2 = robust_min_max(img_gradient_2, 0.001, 0.1)
    # min_value3, max_value3 = robust_min_max(img_gradient_3, 0.001, 0.1)
    # img_gradient_1, img_gradient_2, img_gradient_3 = (img_gradient_1 - min_value1) / (max_value1 - min_value1), \
    #     (img_gradient_2 - min_value2) / (max_value2 - min_value2), (img_gradient_3 - min_value3) / (max_value3 - min_value3)
    
    # img_gradient_1, img_gradient_2, img_gradient_3 = sigmoid_mapping(img_gradient_1, 10, 0.02), \
    #     sigmoid_mapping(img_gradient_2, 10, 0.02), sigmoid_mapping(img_gradient_3, 10, 0.02)
    # img_gradient_ = (img_gradient_1 + img_gradient_2 + img_gradient_3) / 3
    # img_gradient_1, img_gradient_2, img_gradient_3 = sigmoid_mapping2(img_gradient_1, 10, 0.75), \
    #     sigmoid_mapping2(img_gradient_2, 10, 0.5), sigmoid_mapping2(img_gradient_3, 10, 0.25)
    # img_gradient_1, img_gradient_2, img_gradient_3 = sigmoid_mapping2(img_gradient_1, 10, 0.5), \
    #     sigmoid_mapping2(img_gradient_2, 10, 0.3), sigmoid_mapping2(img_gradient_3, 10, 0.1)
    
    # 梯度映射，突出中灰度的区分度
    img_gradient_1, img_gradient_2, img_gradient_3 = sigmoid_mapping3(img_gradient_1, 10), \
         sigmoid_mapping3(img_gradient_2, 10), sigmoid_mapping3(img_gradient_3, 10)
    # 多尺度融合
    img_gradient_ = grad_multi_scale_fusion(img_gradient_1, 0.75) * grad_multi_scale_fusion(img_gradient_2, 0.75) * \
        grad_multi_scale_fusion(img_gradient_3, 0.25)
    img_gradient_ = (img_gradient_ - img_gradient_.min())/(img_gradient_.max() - img_gradient_.min() + 1e-22)
    # 用单像素宽度的梯度替代模糊边缘的宽的梯度
    grad_mask = local_max_gradient(img_gradient_)
    img_gradient_4 = grad_mask * img_gradient_  

    # 扩展梯度
    grad_boundary = boundary4gradient_expand(img_gradient_4, 1e20)
    expanded_grad = img_gradient_4
    region_size = region.shape[2] if region.shape[2] > region.shape[3] else region.shape[3]
    for z in range(region_size):
        expanded_grad = gradient_expand_one_step_boundary(expanded_grad, grad_boundary)

    # # 如果在24个方向维度上的最大值和和值一致，则为噪音
    # max_gradient = torch.max(expanded_grad[0], dim=0).values
    # sum_gradient = torch.sum(expanded_grad[0], dim=0)
    # grad_mask = torch.where((max_gradient == sum_gradient) * (max_gradient != 0), torch.zeros_like(max_gradient), torch.ones_like(max_gradient))
    # expanded_grad = expanded_grad * grad_mask

    _target = torch.sum(expanded_grad[0], dim=0)
    # expanded_grad_ = torch.roll(expanded_grad, shifts=1, dims=[1])
    # expanded_grad_ = expanded_grad_.reshape([1, 3, 8, region.shape[-2], region.shape[-1]])
    # # expanded_grad_ = torch.max(expanded_grad_, dim=1, keepdim=False).values
    # expanded_grad = expanded_grad_[:,1]
    # _target = torch.sum(expanded_grad[0], dim=0)
    _target = (_target - _target.min())/(_target.max() - _target.min() + 1e-8)

    # # 显示结果
    # plt.figure(figsize=(25, 25))
    # plt.subplot(151), plt.imshow(region[0,0], cmap='gray')
    # plt.subplot(152), plt.imshow(torch.sum(img_gradient_4[0], dim=0), cmap='gray')
    # plt.subplot(153), plt.imshow(expanded_grad[0,0], cmap='gray')
    # plt.subplot(154), plt.imshow(expanded_grad[0,6], cmap='gray')
    # plt.subplot(155), plt.imshow(torch.sum(smaller_mask[0], dim=0), cmap='gray')
    # # plt.show()

    # # 显示结果, 创建一个 5x6 的子图布局
    # fig, axes = plt.subplots(6, 4, figsize=(18, 12))  # 可调整 figsize 控制整体大小
    # for i in range(4):
    #     for j in range(6):
    #         ax = axes[j, i]
    #         ax.imshow(expanded_grad[0, i * 6 + j], cmap='gray')
    #         ax.set_title(f"{(i * 6 + j)*15}")
    # # plt.show()

    return _target


def expand_and_contract_mask(mask, d1, d2):
    """
    对目标mask的边缘进行向外和向内扩展。
    
    参数:
        mask (torch.Tensor): 输入的目标mask，形状为 [1, 1, H, W]。
        d1 (int): 向外扩展的像素数。
        d2 (int): 向内收缩的像素数。
    
    返回:
        torch.Tensor: 处理后的mask，形状与输入相同，取值为0或1。
    """
    
    # 使用最大池化实现向外扩展
    kernel_size_d1 = 2 * d1 + 1
    expanded_mask = F.max_pool2d(mask.float(), kernel_size=kernel_size_d1, stride=1, padding=d1)
    
    # 使用腐蚀操作（最小池化）实现向内收缩
    kernel_size_d2 = 2 * d2 + 1
    contracted_mask = -F.max_pool2d(-mask.float(), kernel_size=kernel_size_d2, stride=1, padding=d2)

    # 对结果取高斯模糊，即不产生锐利的mask，而是宽容度更高的
    kernel_size = min(d1-1, d2-1)*2 + 1
    gaussian_kernel_1d = gaussian_kernel(kernel_size, 2)
    gaussian_kernel_ = torch.outer(gaussian_kernel_1d, gaussian_kernel_1d)
    gaussian_kernel_ = gaussian_kernel_.expand(1, 1, kernel_size, kernel_size)
    result_mask = F.conv2d(expanded_mask, gaussian_kernel_, padding=gaussian_kernel_.shape[-1]//2)

    # 取两者的交集：向外扩展的部分与向内收缩的部分
    result_mask = result_mask * (contracted_mask < 1.0)
    # result_mask = (result_mask > 0.5).float()

    if torch.max(result_mask) < 1.0:
        result_mask = generate_gaussian_mask_with_edge_value(mask.shape[-2], mask.shape[-1])
        result_mask = result_mask.unsqueeze(0).unsqueeze(0)
    return result_mask


def proper_region(pred, c1, c2):
    """
    由训练过的模型的预测和点标签的坐标，得到一个合适的区域。
    参数:
        pred (torch.Tensor): 形状为 [H, W]。
    输出：
        s1 (int): 区域的起始高度索引。
        e1 (int): 区域的结束高度索引。
        s2 (int): 区域的起始宽度索引。
        e2 (int): 区域的结束宽度索引。
    """
    initial_size = 32
    half_size = initial_size // 2
    pred_ = F.pad(pred, [half_size, half_size, half_size, half_size], value=0)
    s1 = c1
    e1 = c1 + initial_size
    s2 = c2
    e2 = c2 + initial_size
    mini_size = 2
    extend_size = 3
    # 合适上边界
    for i in range(half_size, mini_size//2, -1):
        s1 = c1 + half_size - i
        if torch.max(pred_[s1, s2:e2]) > 0.1:
            break
    # 下边界
    for i in range(half_size, mini_size//2, -1):
        e1 = c1 + half_size + i
        if torch.max(pred_[e1, s2:e2]) > 0.1:
            break
    # 左边界
    for i in range(half_size, mini_size//2, -1):
        s2 = c2 + half_size - i
        if torch.max(pred_[s1:e1, s2]) > 0.1:
            break
    # 右边界
    for i in range(half_size, mini_size//2, -1):
        e2 = c2 + half_size + i
        if torch.max(pred_[s1:e1, e2]) > 0.1:
            break
    
    s1 = s1 - half_size - extend_size if s1 - half_size - extend_size > 1 else 1
    e1 = e1 - half_size + extend_size if e1 - half_size + extend_size < pred.shape[0] - 2 else pred.shape[0] - 2
    s2 = s2 - half_size - extend_size if s2 - half_size - extend_size > 1 else 1
    e2 = e2 - half_size + extend_size if e2 - half_size + extend_size < pred.shape[1] - 2 else pred.shape[1] - 2
    return (s1, e1, s2, e2)
 

def target_adanptive_filtering(target, img, pred=None, view=False):
    def find_min_above(tensor, threshold):
        filtered = tensor[tensor > threshold]
        if filtered.numel() == 0:
            return None
        return filtered.min().item()
    # 归一化到 0-255 范围内
    min_val = target.min()
    max_val = target.max()
    target_normal = ((target - min_val) / (max_val - min_val) * 255).type(torch.uint8)
    # 过滤梯度扩展的图形结果
    hist, bins = compute_histogram(target_normal)
    # # 为了降低低值像素的巨大数量对平滑曲线的巨大影响，我们直接限制最大值为5
    # limitation = 3
    # hist = torch.where(hist > limitation, torch.ones_like(hist) * limitation, hist)
    hist = hist_mapping(hist, 0.5)
    ## 平滑处理
    smooth_hist = smooth_histogram(hist.numpy(), 3, 3)
    smoother_hist = smooth_histogram(hist.numpy(), 10, 3)    # 更加全局的曲线

    peaks, props = fpk(-smooth_hist, prominence=0.01, width=0.5)
    peaks_2, props_2 = fpk(-smoother_hist, prominence=0.001, width=0.5)

    def proper_peak2(peaks, props, peaks2, props2, pred, target):
        """
        根据波峰的属性，选出range范围内最合适的波峰。
        属性包括：显著性，波峰宽度，靠经第一个波峰（小比例），靠近大波峰, 所靠近的大波峰的宽度，显著性
        参数：
            peaks (list): 波峰的索引列表。
            props (dict): 波峰的属性字典，包括 prominence、left_bases 和 right_bases 等。
            peaks2 (list): 大波峰的索引列表。
            props2 (dict): 大波峰的属性字典，包括 prominence、left_bases 和 right_bases 等。
            pred: numpy.array, 预测的灰度值。
            target: numpy.array, 由传统算法预测的logits
        返回：
            int: 最合适的波峰的索引。
        注意：
            如果波峰的数量为0，则返回-1。
        """
        if len(peaks) == 0:
            return 0
        score = [ 0 for i in range(len(peaks))]

        def mapping_list(L, S=0.3):

            L = np.array(L)
            N = len(L)
            if N == 0:
                return []

            top_n = int(N * S)
            if top_n < 1:
                top_n = 1  # 至少保留一个最大值为1

            max_L = np.max(L)
            min_L = np.min(L)
            lower_bound = 0 if max_L == 0 or (min_L <= 0 and max_L > 0) else min_L / max_L

            # 创建结果数组
            result = np.zeros_like(L, dtype=float)

            # 获取排序索引
            sorted_indices = np.argsort(L)
            rest_indices = sorted_indices[:-top_n]
            top_indices = sorted_indices[-top_n:]

            # 设置 top_n 个最大值为 1
            result[top_indices] = 1.0

            # 对其余元素进行线性映射到 [lower_bound, 1)
            if len(rest_indices) > 0:
                rest_values = L[rest_indices]
                rest_min = rest_values.min()
                rest_max = rest_values.max()
                if rest_min == rest_max:
                    # 所有剩余值相同
                    result[rest_indices] = (1 + lower_bound) / 2
                else:
                    normalized = (rest_values - rest_min) / (rest_max - rest_min)
                    result[rest_indices] = (1 - lower_bound) * normalized + lower_bound

            return result.tolist()
        
        # 波峰宽度
        ratio = 1.0
        width_rank = np.argsort(props['widths'])
        # # print(props['widths'])
        width_score = []
        for i in range(len(score)):
            width_score.append(props['widths'][i]/props['widths'][width_rank[-1]])
        # width_score = mapping_list(width_score)
        for i in range(len(score)):
            score[i] = score[i] + width_score[i] * ratio
        # print(width_score)

        # 显著性
        ratio = 1.0
        prominence_rank = np.argsort(props['prominences'])
        prominence_score = []
        for i in range(len(score)):
            prominence_score.append(props['prominences'][i]/props['prominences'][prominence_rank[-1]])
        # prominence_score = mapping_list(prominence_score)
        for i in range(len(score)):
            score[i] = score[i] + prominence_score[i] * ratio
        # print(prominence_score)

        # if len(peaks2) > 0:
        #     # 寻找最接近的大波峰
        #     def find_closest_elements(array_a, array_b):
        #         """
        #         对 array_a 中每个元素，在 array_b 中找到最接近的元素值和索引。

        #         Args:
        #             array_a (np.ndarray): 形状 (W,), 查询点
        #             array_b (np.ndarray): 形状 (K,), 候选点

        #         Returns:
        #             closest_values (np.ndarray): 形状 (W,), 每个位置是 array_a[i] 在 array_b 中最近的值
        #             closest_indices (np.ndarray): 形状 (W,), 最近值在 array_b 中的索引
        #         """
        #         # 扩展维度，使 array_a: [W, 1], array_b: [1, K]
        #         diff = np.abs(array_a[:, np.newaxis] - array_b[np.newaxis, :])  # [W, K]

        #         # 找到 array_b 中距离最小的索引
        #         closest_indices = np.argmin(diff, axis=1)  # [W]

        #         # 根据索引取出对应的值
        #         closest_values = array_b[closest_indices]

        #         return closest_values, closest_indices
        
        #     values, indices = find_closest_elements(peaks, peaks2)
        #     # 计算与大波峰的靠近性
        #     dist = np.abs(peaks - values)
        #     dist_rank = np.argsort(dist)
        #     ratio = 0.2
        #     dist_score = []
        #     for i in range(len(score)):
        #         dist_score.append((dist[dist_rank[-1]] - dist[i])/dist[dist_rank[-1]])
        #     dist_score = mapping_list(dist_score, 0.1)
        #     for i in range(len(score)):
        #         score[i] = score[i] + dist_score[i] * ratio
        #     # # print(dist_score)

        #     # 计算靠近大波峰的宽度
        #     ratio = 0.4
        #     width2_rank = np.argsort(props2['widths'])
        #     width2_score = []
        #     for i in range(len(score)):
        #         width2_score.append(props2['widths'][indices[i]]/props2['widths'][width2_rank[-1]])
        #     width2_score = mapping_list(width2_score, 0.1)
        #     for i in range(len(score)):
        #         score[i] = score[i] + width2_score[i] * ratio
        #     # # print(width2_score)
            
        #     # 计算靠近大波峰的显著性
        #     ratio = 0.3
        #     prominence2_rank = np.argsort(props2['prominences'])
        #     prominence2_score = []
        #     for i in range(len(score)):
        #         prominence2_score.append(props2['prominences'][indices[i]]/props2['prominences'][prominence2_rank[-1]])
        #     prominence2_score = mapping_list(prominence2_score, 0.1)
        #     for i in range(len(score)):
        #         score[i] = score[i] + prominence2_score[i] * ratio

        # if len(peaks2) > 0:
        #     # 计算靠近大波峰的宽度
        #     width2_rank = np.argsort(props2['widths'])
        #     width2_score = []
        #     for i in range(len(peaks2)):
        #         width2_score.append(props2['widths'][i]/props2['widths'][width2_rank[-1]])
            
        #     # 计算靠近大波峰的显著性
        #     prominence2_rank = np.argsort(props2['prominences'])
        #     prominence2_score = []
        #     for i in range(len(peaks2)):
        #         prominence2_score.append(props2['prominences'][i]/props2['prominences'][prominence2_rank[-1]])

        #     # 计算小波峰属于哪一个大波峰，并计算大波峰的加成。
        #     width2_ratio, prominence2_ratio, dist_ratio = 0.4, 0.4, 0.2
        #     for i in range(len(score)):
        #         score_ = 0.
        #         for j in range(len(peaks2)):
        #             if peaks[i] >= props2['left_bases'][j] and peaks[i] <= props2['right_bases'][j]:
        #                 radius = max(props2['right_bases'][j] - peaks2[j], peaks2[j] - props2['left_bases'][j])
        #                 dist_score = 1 - abs(peaks[i] - peaks2[j]) / radius
        #                 score_ = width2_score[j] * width2_ratio + prominence2_score[j] * prominence2_ratio + dist_score * dist_ratio
        #                 break
        #         if score_ == 0:
        #             score_ = 0.5
        #         score[i] += score_ 
        #     # # print(prominence2_score)
        if pred is not None:
            # # 与预测的像素数靠进行
            # target_pixel_num = np.sum((pred.numpy() > 0.1).astype(np.float32))
            # pixel_num_diff = [ 0 for i in range(len(peaks))]
            # for i in range(len(score)):
            #     pixel_num_diff[i] = np.sum(smooth_hist[peaks[i]:]) - target_pixel_num
            # pixel_num_diff_rank = np.argsort(np.abs(pixel_num_diff))
            # ratio = 1.0
            # for i in range(len(score)):
            #     score[pixel_num_diff_rank[i]] = score[pixel_num_diff_rank[i]] + (len(score)-i-1) * ratio
            # 与深度学习模型预测的形状进行比较，取最大的iou对应的波谷。
            pred_mask = (pred > 0.1).astype(np.float32)
            iou_score_ = [ 0 for i in range(len(peaks))]
            for i in range(len(score)):
                target_mask = (target > peaks[i]).astype(np.float32)
                iou_score_[i] = iou_score(pred_mask, target_mask)
            # iou_score_ = mapping_list(iou_score_, 0.1)
            ratio = 4.0
            for i in range(len(score)):
                score[i] = score[i] + iou_score_[i] * ratio
            # print(iou_score_)

            # 边缘部分清洁性
            close_scores = []
            for i in range(len(peaks)):
                target_ = (target > peaks[i]).astype(np.float32)
                close_score = object_closed_score(torch.tensor(target_), 8)
                close_scores.append(close_score)
            close_scores = mapping_list(close_scores)
            ratio = 1
            for i in range(len(score)):
                score[i] = score[i] + close_scores[i] * ratio
            # print(close_scores)

            # 高灰度值一体性
            def calculate_spatial_discontinuity(mask, connectivity=2):
                """
                输入:
                    mask: [H, W] 二维数组，模型输出
                    connectivity: 1 表示四邻域，2 表示八邻域

                输出:
                    discontinuity_score: 空间不连续性得分 (数值越大越不连续)
                """
                # 设置结构元素
                if connectivity == 1:
                    structure = np.array([[0,1,0],
                                        [1,1,1],
                                        [0,1,0]])
                elif connectivity == 2:
                    structure = np.ones((3, 3))
                else:
                    raise ValueError("connectivity must be 1 or 2")

                # Step 3: 标记连通区域
                labeled_array, num_objects = ndlabel(mask, structure=structure)

                if num_objects == 0:
                    return 0.0  # 没有高值区域，没有不连续性

                # Step 4: 计算每个区域的加权像素值总和（权重为像素值）
                sums = ndimage.sum(mask, labeled_array, index=np.arange(1, num_objects + 1))

                # Step 5: 归一化权重分布（变成概率分布）
                total_weight = np.sum(sums)
                if total_weight == 0:
                    return 0.0

                probs = sums / total_weight

                # Step 6: 计算香农熵作为不连续性指标（越高越不连续）
                entropy = -np.sum(probs * np.log(probs + 1e-10))  # 加小量避免 log(0)

                # 可选：结合区域数与熵综合评分
                # 不连续性 = 区域数 × 熵
                discontinuity_score = num_objects * entropy

                return discontinuity_score

            integirty_scores = []
            for i in range(len(peaks)):
                target_ = (target > peaks[i]).astype(np.float32)
                integirty_score = calculate_spatial_discontinuity(target_)
                integirty_scores.append(-integirty_score)   # 为了适配mapping函数与将不连接转化为连接得分，添加负号
            integirty_scores = mapping_list(integirty_scores, 0.1)
            ratio = 2.0
            for i in range(len(score)):
                score[i] = score[i] + integirty_scores[i] * ratio
            # print(integirty_scores)

            # 高灰度值保留性
            highval_keeping_scores = []
            for i in range(len(peaks)):
                target_ = (target > peaks[i]) * target
                highval_keeping_score = target_.sum()
                highval_keeping_scores.append(highval_keeping_score)
            highval_keeping_scores = mapping_list(highval_keeping_scores)
            ratio = 1
            for i in range(len(score)):
                score[i] = score[i] + highval_keeping_scores[i] * ratio
            # print(highval_keeping_scores)

        # print('score', score)
        peaks_idx = np.argmax(score)
        return peaks_idx

    peak_idx = proper_peak2(peaks, props, peaks_2, props_2, pred.numpy(), target_normal.numpy())
    threshold = peaks[peak_idx]

    # def compute_local_thresholds_torch(logits: torch.Tensor,
    #                                mask: torch.Tensor,
    #                                window_size: int = 5,
    #                                metric: str = 'accuracy') -> torch.Tensor:
    #     """
    #     使用 PyTorch 实现基于局部窗口的最优阈值计算，支持 GPU 加速。
        
    #     参数:
    #         logits: [H, W], 浮点型预测分数
    #         mask: [H, W], 整型标签 (0 或 1)
    #         window_size: 局部窗口大小（必须为奇数）
    #         metric: 评估指标，'accuracy' 或 'f1'
            
    #     返回:
    #         thresholds: [H, W], 每个位置的最佳阈值
    #     """
    #     assert window_size % 2 == 1, "window_size must be odd"
    #     device = logits.device
    #     dtype = logits.dtype
    #     H, W = logits.shape

    #     pad = window_size // 2
    #     # 填充边缘以保持输出尺寸一致
    #     logits_padded = F.pad(logits.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='reflect').squeeze()
    #     mask_padded = F.pad(mask.unsqueeze(0).unsqueeze(0).float(), (pad, pad, pad, pad), mode='reflect').squeeze().bool()

    #     # 提取局部窗口：[H, W, win, win]
    #     windows = logits_padded.unfold(0, window_size, 1).unfold(1, window_size, 1)
    #     mask_windows = mask_padded.unfold(0, window_size, 1).unfold(1, window_size, 1)

    #     # 展平每个窗口为一维：[H*W, win*win]
    #     flat_windows = windows.contiguous().view(-1, window_size * window_size)
    #     flat_masks = mask_windows.contiguous().view(-1, window_size * window_size)

    #     B = flat_windows.size(0)  # 总共 H*W 个窗口

    #     # 收集所有可能的候选阈值（所有窗口中的唯一中点）
    #     all_candidates = []
    #     for i in range(B):
    #         vals = torch.unique(flat_windows[i])
    #         if len(vals) > 1:
    #             mid_points = (vals[:-1] + vals[1:]) / 2
    #             all_candidates.append(mid_points)
    #     all_candidates = torch.cat(all_candidates)
    #     unique_candidates = torch.sort(torch.unique(all_candidates))[0]

    #     # 初始化最佳得分和最佳阈值
    #     best_scores = -torch.inf * torch.ones(B, device=device)
    #     thresholds = torch.zeros(B, device=device, dtype=dtype)

    #     # 遍历每个候选阈值
    #     for th in unique_candidates:
    #         pred = (flat_windows > th)  # shape: [B, win^2]

    #         if metric == 'accuracy':
    #             score = (pred == flat_masks).float().mean(dim=1)
    #         elif metric == 'f1':
    #             tp = (pred & (flat_masks == 1)).sum(dim=1).float()
    #             fp = (pred & (flat_masks == 0)).sum(dim=1).float()
    #             fn = (~pred & (flat_masks == 1)).sum(dim=1).float()

    #             precision = tp / (tp + fp + 1e-8)
    #             recall = tp / (tp + fn + 1e-8)
    #             score = 2 * precision * recall / (precision + recall + 1e-8)
    #         else:
    #             raise ValueError(f"Unsupported metric: {metric}")

    #         improved = score > best_scores
    #         thresholds[improved] = th
    #         best_scores[improved] = score[improved]

    #     # 处理没有找到阈值的情况（如所有值都相同）
    #     no_th = torch.isnan(best_scores)
    #     thresholds[no_th] = 255

    #     return thresholds.view(H, W)

    # def compute_local_thresholds(logits: np.ndarray, mask: np.ndarray, window_size: int):
    #     """
    #     Args:
    #         logits (np.ndarray): shape [H, W], model output logits.
    #         mask (np.ndarray): shape [H, W], binary mask with values in {0, 1}.
    #         window_size (int): size of local window.

    #     Returns:
    #         thresholds (np.ndarray): shape [H, W], threshold for each pixel.
    #     """
    #     assert logits.shape == mask.shape
    #     assert window_size % 2 == 1, "Window size must be odd"
        
    #     H, W = logits.shape
    #     pad = window_size // 2
        
    #     # Convert to tensors
    #     logits_tensor = torch.tensor(logits, dtype=torch.float32)
    #     mask_tensor = torch.tensor(mask, dtype=torch.float32)

    #     # Pad the logits and mask for extracting windows at borders
    #     logits_padded = torch.nn.functional.pad(logits_tensor, (pad, pad, pad, pad), mode='constant', value=0)
    #     mask_padded = torch.nn.functional.pad(mask_tensor, (pad, pad, pad, pad), mode='constant', value=0)

    #     # Extract local windows: shape [H, W, window_size, window_size]
    #     local_windows = extract_local_windows(logits_padded, window_size)  # Assume this function is defined
    #     mask_windows = extract_local_windows(mask_padded, window_size)

    #     # Initialize thresholds tensor
    #     thresholds = torch.zeros_like(logits_tensor)

    #     for i in range(H):
    #         for j in range(W):
    #             current_window = local_windows[i, j]  # shape [ws, ws]
    #             current_mask = mask_windows[i, j]      # shape [ws, ws]
    #             current_threshold = current_window[pad, pad]  # center pixel as candidate threshold

    #             # Binary prediction using current threshold
    #             pred = (current_window >= current_threshold).float()

    #             # Compare prediction with ground truth mask
    #             correct = (pred == current_mask).float()
    #             incorrect = (pred != current_mask).float()

    #             correct_count = correct.sum().item()
    #             incorrect_count = incorrect.sum().item()

    #             if correct_count > incorrect_count:
    #                 thresholds[i, j] = current_threshold
    #             else:
    #                 # 可选策略：保留原值 / 设为 None / 设为 inf 表示无效
    #                 thresholds[i, j] = float('nan')  # 或者 current_threshold 不变，根据需求决定

    #     return thresholds.numpy()

    # # 通过深度学习模型预测得到局部阈值
    # pred = pred > 0.1
    # threshold_dl = compute_local_thresholds(target_normal, pred, 3)
    # threshold_dl = torch.tensor(threshold_dl)
    # # threshold_dl = gaussian_blurring_2D(threshold_dl_, 3, 2.0)
    # print(threshold_dl.int())

    # # 通过全局方式得到的局部阈值
    # peak_idx = proper_peak2(peaks, props, peaks_2, props_2, pred.numpy(), target_normal.numpy())
    # threshold = peaks[peak_idx]
    # filtered_target = target_normal > threshold
    # threshold_m = compute_local_thresholds_torch(target_normal, filtered_target, 3, metric='f1')

    # def intensity_mapping(tensor, threshold):
    #     alpha1 = 2 / min(threshold, 255-threshold)
    #     print('alpha1', alpha1)
    #     logits = gradient_mapping(tensor, alpha1, 0.1, threshold)
    #     return logits

    # logits = intensity_mapping(target_normal.float(), threshold)
    # # print(logits)
    # # target = map_tensor(target) 
    # logits = logits.numpy()
    # img = img.numpy()
    # filtered_target2 = apply_crf(logits, img[0,0], 10)
    # filtered_target2 = torch.tensor(filtered_target2)
    filtered_target = target_normal * (target_normal > threshold)
    filtered_target = filtered_target / 255.

    # # 绘制直方图
    # fig = plt.figure(figsize=(20, 5))
    
    # # 原始图像
    # plt.subplot(1, 3, 1)
    # plt.imshow(target, cmap='gray')
    
    # # 直方图 
    # plt.subplot(1, 3, 2)
    # # plt.imshow(target_normal * (target_normal > threshold_dl) / 255, cmap='gray', vmax=1.0, vmin=0.0)
    # # plt.title('Filtered Image')
    # plt.bar(bins, hist, color='blue', alpha=0.7, label='Histogram')
    # plt.plot(bins, smooth_hist, color='orange', label='Smoothed Histogram')
    # plt.plot(bins, smoother_hist, color='green', label='Smoothed Histogram _2')
    # if peaks is not None:
    #     for i in peaks:
    #         plt.axvline(x=i, color='red', linestyle='--')
    # if peaks_2 is not None:
    #     for i in peaks_2:
    #         plt.axvline(x=i, color='cyan', linestyle='--')
    # plt.axvline(x=threshold, color='purple', linestyle='--') 
    # plt.legend()
    # plt.title('Brightness Histogram')
    # plt.xlabel('Brightness Level')
    # plt.ylabel('Pixel Count')
    # plt.ylim(0, 10)  # 设置bottom和top为你想要的y轴范围
    
    # # 过滤后的图像
    # plt.subplot(1, 3, 3)
    # plt.imshow(filtered_target, cmap='gray', vmax=1.0, vmin=0.0)
    # plt.title('Filtered Image')

    # plt.show()

    
    return filtered_target


def target_adanptive_filtering_v2(target, img, pred=None, view=False):
    def find_min_above(tensor, threshold):
        filtered = tensor[tensor > threshold]
        if filtered.numel() == 0:
            return None
        return filtered.min().item()
    # 归一化到 0-255 范围内
    min_val = target.min()
    max_val = target.max()
    target_normal = ((target - min_val) / (max_val - min_val) * 255).type(torch.uint8)
    # 过滤梯度扩展的图形结果
    hist, bins = compute_histogram(target_normal)
    # # 为了降低低值像素的巨大数量对平滑曲线的巨大影响，我们直接限制最大值为5
    # limitation = 3
    # hist = torch.where(hist > limitation, torch.ones_like(hist) * limitation, hist)
    hist = hist_mapping(hist, 0.5)
    ## 平滑处理
    smooth_hist = smooth_histogram(hist.numpy(), 3, 3)
    smoother_hist = smooth_histogram(hist.numpy(), 10, 3)    # 更加全局的曲线

    peaks, props = fpk(-smooth_hist, prominence=0.01, width=0.5)
    peaks_2, props_2 = fpk(-smoother_hist, prominence=0.001, width=0.5)

    def peaks_probability(peaks, props, pred, target):
        """
        根据波峰的属性，选出range范围内最合适的波峰。
        属性包括：显著性，波峰宽度，靠经第一个波峰（小比例），靠近大波峰, 所靠近的大波峰的宽度，显著性
        参数：
            peaks (list): 波峰的索引列表。
            props (dict): 波峰的属性字典，包括 prominence、left_bases 和 right_bases 等。
            peaks2 (list): 大波峰的索引列表。
            props2 (dict): 大波峰的属性字典，包括 prominence、left_bases 和 right_bases 等。
            pred: numpy.array, 预测的灰度值。
            target: numpy.array, 由传统算法预测的logits
        返回：
            int: 最合适的波峰的索引。
        注意：
            如果波峰的数量为0，则返回-1。
        """
        if len(peaks) == 0:
            return 0
        score = [ 0 for i in range(len(peaks))]

        def mapping_list(L, S=0.3):

            L = np.array(L)
            N = len(L)
            if N == 0:
                return []

            top_n = int(N * S)
            if top_n < 1:
                top_n = 1  # 至少保留一个最大值为1

            max_L = np.max(L)
            min_L = np.min(L)
            lower_bound = 0 if max_L == 0 or (min_L <= 0 and max_L > 0) else min_L / max_L

            # 创建结果数组
            result = np.zeros_like(L, dtype=float)

            # 获取排序索引
            sorted_indices = np.argsort(L)
            rest_indices = sorted_indices[:-top_n]
            top_indices = sorted_indices[-top_n:]

            # 设置 top_n 个最大值为 1
            result[top_indices] = 1.0

            # 对其余元素进行线性映射到 [lower_bound, 1)
            if len(rest_indices) > 0:
                rest_values = L[rest_indices]
                rest_min = rest_values.min()
                rest_max = rest_values.max()
                if rest_min == rest_max:
                    # 所有剩余值相同
                    result[rest_indices] = (1 + lower_bound) / 2
                else:
                    normalized = (rest_values - rest_min) / (rest_max - rest_min)
                    result[rest_indices] = (1 - lower_bound) * normalized + lower_bound

            return result.tolist()
        
        # 波峰宽度
        ratio = 1.0
        width_rank = np.argsort(props['widths'])
        # # print(props['widths'])
        width_score = []
        for i in range(len(score)):
            width_score.append(props['widths'][i]/props['widths'][width_rank[-1]])
        # width_score = mapping_list(width_score)
        for i in range(len(score)):
            score[i] = score[i] + width_score[i] * ratio
        # print(width_score)

        # 显著性
        ratio = 1.0
        prominence_rank = np.argsort(props['prominences'])
        prominence_score = []
        for i in range(len(score)):
            prominence_score.append(props['prominences'][i]/props['prominences'][prominence_rank[-1]])
        # prominence_score = mapping_list(prominence_score)
        for i in range(len(score)):
            score[i] = score[i] + prominence_score[i] * ratio
        # print(prominence_score)

        if pred is not None:
            # 与深度学习模型预测的形状进行比较，取最大的iou对应的波谷。
            pred_mask = (pred > 0.1).astype(np.float32)
            iou_score_ = [ 0 for i in range(len(peaks))]
            for i in range(len(score)):
                target_mask = (target > peaks[i]).astype(np.float32)
                iou_score_[i] = iou_score(pred_mask, target_mask)
            # iou_score_ = mapping_list(iou_score_, 0.1)
            ratio = 2.0
            for i in range(len(score)):
                score[i] = score[i] + iou_score_[i] * ratio
            # print(iou_score_)

            # 边缘部分清洁性
            close_scores = []
            for i in range(len(peaks)):
                target_ = (target > peaks[i]).astype(np.float32)
                close_score = object_closed_score(torch.tensor(target_), 8)
                close_scores.append(close_score)
            # close_scores = mapping_list(close_scores, 0.01)
            max_close_score = np.max(close_scores) 
            for i in range(len(close_scores)):
                close_scores[i] = close_scores[i]/max_close_score
            ratio = 1.0
            for i in range(len(score)):
                score[i] = score[i] + close_scores[i] * ratio
            # print(close_scores)

            # 高灰度值一体性
            def calculate_spatial_discontinuity(mask, connectivity=2):
                """
                输入:
                    mask: [H, W] 二维数组，模型输出
                    connectivity: 1 表示四邻域，2 表示八邻域

                输出:
                    discontinuity_score: 空间不连续性得分 (数值越大越不连续)
                """
                # 设置结构元素
                if connectivity == 1:
                    structure = np.array([[0,1,0],
                                        [1,1,1],
                                        [0,1,0]])
                elif connectivity == 2:
                    structure = np.ones((3, 3))
                else:
                    raise ValueError("connectivity must be 1 or 2")

                # Step 3: 标记连通区域
                labeled_array, num_objects = ndlabel(mask, structure=structure)

                if num_objects == 0:
                    return 0.0  # 没有高值区域，没有不连续性

                # Step 4: 计算每个区域的加权像素值总和（权重为像素值）
                sums = ndimage.sum(mask, labeled_array, index=np.arange(1, num_objects + 1))

                # Step 5: 归一化权重分布（变成概率分布）
                total_weight = np.sum(sums)
                if total_weight == 0:
                    return 0.0

                probs = sums / total_weight

                # Step 6: 计算香农熵作为不连续性指标（越高越不连续）
                entropy = -np.sum(probs * np.log(probs + 1e-10))  # 加小量避免 log(0)

                # 可选：结合区域数与熵综合评分
                # 不连续性 = 区域数 × 熵
                discontinuity_score = num_objects * entropy

                return discontinuity_score

            integirty_scores = []
        for i in range(len(peaks)):
            target_ = (target > peaks[i]).astype(np.float32)
            integirty_score = calculate_spatial_discontinuity(target_)
            integirty_scores.append(-integirty_score)   # 为了适配mapping函数与将不连接转化为连接得分，添加负号
        # integirty_scores = mapping_list(integirty_scores, 0.01)
        max_integirty_score, min_integirty_score = np.max(integirty_scores), np.min(integirty_scores)
        for i in range(len(integirty_scores)):
            integirty_scores[i] = (integirty_scores[i] - min_integirty_score) / (max_integirty_score - min_integirty_score + 1e-11)
        ratio = 2.0
        for i in range(len(score)):
            score[i] = score[i] + integirty_scores[i] * ratio
        # print(integirty_scores)

        # 高灰度值保留性
        highval_keeping_scores = []
        for i in range(len(peaks)):
            target_ = (target > peaks[i]) * target
            highval_keeping_score = target_.sum()
            highval_keeping_scores.append(highval_keeping_score)
        max_highval_keeping_score = np.max(highval_keeping_scores)
        for i in range(len(highval_keeping_scores)):
            highval_keeping_scores[i] = highval_keeping_scores[i]/max_highval_keeping_score
        ratio = 1.0
        for i in range(len(score)):
            score[i] = score[i] + highval_keeping_scores[i] * ratio
            # print(highval_keeping_scores)

        # print('score', score)
        sorted_score_idx = np.argsort(score)
        probs = []

        for i in range(len(score)):
            if i <= sorted_score_idx[-1]:
                prob = (score[i] - score[sorted_score_idx[0]]) / (score[sorted_score_idx[-1]] - score[sorted_score_idx[0]])
            else:
                # prob = 1 + (score[i] - score[sorted_score_idx[0]]) / (score[sorted_score_idx[-1]] - score[sorted_score_idx[0]])
                prob = 1.
            probs.append(prob)
            # peak_idx.append(i)
        # peak_idx = []
        # for i in sorted_score_idx[-2:]:
        #     if i <= sorted_score_idx[-1]:
        #         prob = (score[i] - score[sorted_score_idx[0]]) / (score[sorted_score_idx[-1]] - score[sorted_score_idx[0]])
        #     else:
        #         prob = 1 + (score[sorted_score_idx[-1]] - score[i]) / (score[sorted_score_idx[-1]] - score[sorted_score_idx[0]])
        #     probs.append(prob)
        #     peak_idx.append(i)
        # return probs, peak_idx
        return probs

    peak_probs = peaks_probability(peaks, props, pred.numpy(), target_normal.numpy())

    filtered_target = torch.zeros_like(target_normal)
    for i in range(len(peaks)):
        threshold_filtered_area = (target_normal > peaks[i]).float() * peak_probs[i]
        filtered_target = torch.where(threshold_filtered_area > filtered_target, threshold_filtered_area, filtered_target)
        # filtered_target = torch.where(threshold_filtered_area > 0., threshold_filtered_area, filtered_target)
        # print(peaks[i], peak_probs[i])
    # peaks_idx = peak_probs.index(1.)
    # filtered_target = (target_normal > peaks[peaks_idx]).float() * target_normal
    # filtered_target = filtered_target/ 255.
    # threshold = peaks[peaks_idx]
    threshold = 0.
 
    # # 绘制直方图
    # fig = plt.figure(figsize=(15, 5))
    
    # # 原始图像
    # plt.subplot(1, 3, 1)
    # plt.imshow(target, cmap='gray')
    
    # # 直方图 
    # plt.subplot(1, 3, 2)
    # # plt.imshow(target_normal * (target_normal > threshold_dl) / 255, cmap='gray', vmax=1.0, vmin=0.0)
    # # plt.title('Filtered Image')
    # plt.bar(bins, hist, color='blue', alpha=0.7, label='Histogram')
    # plt.plot(bins, smooth_hist, color='orange', label='Smoothed Histogram')
    # plt.plot(bins, smoother_hist, color='green', label='Smoothed Histogram _2')
    # if peaks is not None:
    #     for i in peaks:
    #         plt.axvline(x=i, color='red', linestyle='--')
    # if peaks_2 is not None:
    #     for i in peaks_2:
    #         plt.axvline(x=i, color='cyan', linestyle='--')
    # plt.axvline(x=threshold, color='purple', linestyle='--') 
    # plt.legend()
    # plt.title('Brightness Histogram')
    # plt.xlabel('Brightness Level')
    # plt.ylabel('Pixel Count')
    # plt.ylim(0, 10)  # 设置bottom和top为你想要的y轴范围
    
    # # 过滤后的图像
    # plt.subplot(1, 3, 3)
    # plt.imshow(filtered_target, cmap='gray')
    # plt.title('Filtered Image')

    # plt.show()

    if view:
        return filtered_target, (bins, hist, smooth_hist, smoother_hist, peaks, peaks_2, threshold)

    return filtered_target, ()


def save_pesudo_label(pesudo_l, save_path, name_path):
    """
    将伪标签保存到指定路径。

    参数:
        pesudo_l (list): 伪标签列表。
        save_path (str): 保存路径。
        name_path (str): 标签名称来源的地址。
    """
    names = os.listdir(name_path)
    print(f"pesudo_l num: {len(pesudo_l)}")
    print(f"names num: {len(names)}")
    for i, pesudo in enumerate(pesudo_l):
        pesudo_label = np.zeros((256, 256), dtype=np.uint8)
        # pesudo_label = np.zeros((512, 512), dtype=np.uint8)
        for j, target_info in enumerate(pesudo):
            if len(target_info) < 1:
                continue
            target = target_info["target"]
            # 归一化到 0-255 范围内
            min_val = target.min()
            max_val = target.max()
            target = ((target - min_val) / (max_val - min_val + 1e-8) * 255).type(torch.uint8)
            target = np.array(target.cpu())
            s1, e1, s2, e2 = target_info["idx"]
            pesudo_label[s1:e1, s2:e2] = target
        # 保存伪标签
        pesudo_label = Image.fromarray(pesudo_label, mode='L')  # 'L' 表示灰度模式
        pesudo_label.save(save_path + '/' + names[i])
        # print(names[i], 'is saved')


def smooth_and_scale_mask(mask, a=0.1, b=0.9, sigma=None, kernel_size=None):
    """
    Args:
        mask (torch.Tensor): shape [H, W], 值接近 0.0 或 1.0。
        a (float): 输出的最小值
        b (float): 输出的最大值
        sigma (float): 高斯模糊的标准差
    
    Returns:
        torch.Tensor: 处理后的 mask，shape [H, W]，值在 [a, b] 范围内
    """
    # 确保输入是 float 类型
    mask = mask.float()
    if sigma is not None:
        # mask_ = dilate_mask(mask, 1)
        mask_ = mask

        # 高斯模糊
        mask_smooth = gaussian_blurring_2D(mask_, kernel_size=kernel_size, sigma=sigma)
        mask = mask_smooth  # [H, W]

    # 线性变换到 [a, b]
    x_min, x_max = mask.min(), mask.max()

    # 使用线性映射：x' = a + (x - x_min) * (b - a) / (x_max - x_min)
    x_scaled = a + (mask - x_min) * (b - a) / (x_max - x_min + 1e-8)  # 加上小数防止除零

    return x_scaled


def fusion_tm_dl(target, pred, alpha=0.5, beta=0.75, sigma=0.9):
    """
    融合固定算法和深度学习模型所产生的伪标签。
    参数:
    target (torch.Tensor): 算法输出的伪标签。
    pred (torch.Tensor): 模型输出的预测， 仅包含。
    """
    # aux_pred = smooth_and_scale_mask(pred, 0.5, 1.0, sigma=2.0, kernel_size=3)
    # aux_target = smooth_and_scale_mask(target, 0.25, 1.0)
    # aux_pred = smooth_and_scale_mask(pred, alpha, 1.0, sigma=sigma, kernel_size=3)
    aux_pred = smooth_and_scale_mask(pred, alpha, 1.0)
    aux_target = smooth_and_scale_mask(target, beta, 1.0)
    fusion = aux_pred * aux_target
    min_val, max_val = fusion.min(), fusion.max()
    return (fusion - min_val) / (max_val - min_val + 1e-8)


def fusion_tm_dl_v2(target, pred):
    """
    融合固定算法和深度学习模型所产生的伪标签。
    原则是通过遍历target和pred两个伪标签候选的决策权重，形成不pred不一致且与target中为1的区域尽量不一致的新的伪标签。
    参数:
    target (torch.Tensor): 算法输出的伪标签。
    pred (torch.Tensor): 模型输出的预测， 仅包含。
    """
    target_mask = (target >= 1.).float()
    pred_mask = (pred > 0.1).float()
    if torch.max(pred_mask) <= 0.1:
        return target_mask
    IoU = iou_score(target_mask.numpy(), pred_mask.numpy())
    if IoU > 0.9:
        return target_mask
    
    # 四舍五入到小数点后 4 位再取唯一值
    unique_target = torch.unique(torch.round(target * 10000) / 10000)   # 默认返回排序后得结果
    if unique_target.shape[0] <= 2:  # 确保至少有 2 个值
        target_lower_limit = unique_target[1] * 0.8
    else:
        target_lower_limit = 2 * unique_target[1] - unique_target[2]
        target_lower_limit = target_lower_limit if target_lower_limit > 0 else 0
    unique_target[0] = target_lower_limit
    # print('unique_target: ', unique_target)

    #
    def fusion_score(target_mask, pred_mask, filtered_mask):
        """
        通过计算filtered_mask是否达到与pred不同，与target尽量不同得效果
        Returns:
            fusion_score, float
        """
        target_only = (1-pred_mask) * target_mask
        pred_only = (1-target_mask) * pred_mask
        inter_area = target_mask * pred_mask
        target_only_iou = iou_score(target_only.numpy(), filtered_mask.numpy())
        pred_only_iou = iou_score(pred_only.numpy(), filtered_mask.numpy())
        inter_area_iou = iou_score(inter_area.numpy(), filtered_mask.numpy())

        score = 0
        if torch.min(pred_mask[target_mask == 1]) == 1:
            score -= 1.0
        elif torch.min(target_mask[pred_mask == 1]) == 1:
            score -= 0.5
        elif target_only_iou == 0:
            score -= 1.0
        elif pred_only_iou == 0:
            score -= 0.5

        score += 0.3 * target_only_iou + 0.3 * pred_only_iou + 0.4 * inter_area_iou
        # print(target_only_iou, pred_only_iou, inter_area_iou, score)
        return score

    scores = []
    filtered_areas = []
    target_ = torch.clamp_min(target, target_lower_limit)
    for i in range(1, unique_target.shape[0]):
        for j in range(i, unique_target.shape[0]):
            pred_lower_limit = (unique_target[i] + unique_target[i-1])/(2 * unique_target[j])
            pred_ = pred_lower_limit + pred_mask * (1 - pred_lower_limit)
        
            fusion = pred_ * target_

            unique_fusion = torch.unique(torch.round(fusion * 10000) / 10000)
            # print('unique_fusion: ',unique_fusion)

            for k in range(1, unique_fusion.shape[0]):
                filtered_area = (fusion >= unique_fusion[k]).float()
                score = fusion_score(target_mask, pred_mask, filtered_area)
                scores.append(score)
                filtered_areas.append(filtered_area)
    # print(scores)
    
    max_score_idx = np.argmax(scores)
    
    return filtered_areas[max_score_idx]


def generate_gaussian_mask_with_edge_value(H, W, edge_value=0.01):
    """
    生成一个从中心向边缘呈高斯分布的 [H, W] tensor，
    并确保边缘点的值不低于给定的 edge_value。

    Args:
        H (int): 高度
        W (int): 宽度
        edge_value (float): 边缘点的最小高斯响应值，默认 0.01

    Returns:
        torch.Tensor: 形状为 [H, W] 的高斯 mask
    """
    # 中心坐标
    cy, cx = H / 2.0, W / 2.0

    # 距离中心最远的角点的距离平方
    max_dist_sq = (cy)**2 + (cx)**2

    # 根据 edge_value 计算 sigma
    sigma = math.sqrt(max_dist_sq / (-2 * math.log(edge_value)))

    # 创建网格坐标
    grid_x = torch.arange(W, dtype=torch.float32)
    grid_y = torch.arange(H, dtype=torch.float32)
    x, y = torch.meshgrid(grid_y, grid_x, indexing='ij')  # [H, W]

    # 高斯分布公式
    dist_sq = (x - cy) ** 2 + (y - cx) ** 2
    gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))

    # 归一化到 [0, 1]
    gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

    return gaussian


def label_evolution_V1(model_path, training_dataset_path):
    """
    进行标签进化的主函数。

    参数:
        model_path (str): 模型文件的路径。
        training_dataset_path (str): 训练数据集的路径。
    """
    # 加载模型
    ## cfg file
    with open("./cfg.yaml") as f:
        cfg = yaml.safe_load(f)
    # model = attenMultiplyUNet_withloss(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNANet_withloss(1, 
                    input_channels=3, 
                    block=Res_CBAM_block,
                    num_blocks=[2, 2, 2, 2],
                    nb_filter=[16, 32, 64, 128, 256],
                    deep_supervision=True,
                    grad_loss=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    # 加载数据集
    dataset = NUDTDataset(training_dataset_path, "train", 256, pt_label=True, pesudo_label=True, augment=False, turn_num=1.16, cfg=cfg)
    # dataset = IRSTD1kDataset(training_dataset_path, "train", 512, pt_label=True, pesudo_label=True, augment=False, turn_num='0.21', cfg=cfg)
    dataloader = Data.DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)

    pesudo_l = []
    img_idx = 0
    preds = []
    # 遍历数据集进行标签进化
    for i, (image, pt_label, pesudo_label) in enumerate(dataloader):
        # 预测
        with torch.no_grad(): 
            image = image.to(device)
            image_ = image.repeat(1, 3, 1, 1)  # for DNANet
            # image_ = TF.resize(image_, (256, 256), antialias=True)  # for IRSTD-1K Dataset
            # pred, _, _, _, _ = model.net(image)
            pred, _ = model.net(image_)
        pred = pred[-1]
        # pred = F.interpolate(pred, scale_factor=2, mode='bilinear', align_corners=True)
        preds.append(pred.cpu())

    for i, (image, pt_label, pesudo_label) in enumerate(dataloader):
        # 预测
        pred = preds[i]
        pred_ = (pred > 0.1).float()
        # 截出点标签的区域
        indices = torch.where(pt_label > 1e-4)

        pesudo_l = pesudo_l + [[] for j in range(image.shape[0])]
        d1,d2 = 3, 8
        for b, _, c1, c2 in zip(*indices):
            s1, e1, s2, e2 = proper_region(pred_[b, 0] + pesudo_label[b, 0], c1, c2)
            
            region = image[b:b+1, :, s1:e1, s2:e2]
            target_ = gradient_expand_one_size(region)
            # 优化伪标签
            target = fusion_tm_dl(target_, pred_[b, 0, s1:e1, s2:e2], 0.6, 0.2)
            # target = target_
            final_target_ = target_adanptive_filtering_v2(target, region, pred[b, 0, s1:e1, s2:e2])
            # 审查，新的伪标签和上一轮伪标签的差距在一定范围内，若差距过大，则还是使用上一轮的伪标签
            final_target = examine_iou(final_target_, pesudo_label[b, 0, s1:e1, s2:e2], iou_treshold=0.5)
            # final_target = sum_val_filter(target, 10)
            # 保存结果
            img_idx = i * 16 + b
            # target_info = {"target": 0}
            target_info = {"target": final_target, "idx": (s1, e1, s2, e2)}
            pesudo_l[img_idx].append(target_info)
            # 显示结果
            plt.figure(figsize=(30, 6))
            plt.subplot(151), plt.imshow(region[0,0], cmap='gray', vmax=1., vmin=0.)
            plt.subplot(152), plt.imshow(target_, cmap='gray', vmax=1., vmin=0.)
            plt.subplot(153), plt.imshow(target, cmap='gray', vmax=1., vmin=0.)
            plt.subplot(154), plt.imshow(final_target, cmap='gray', vmax=1., vmin=0.)
            plt.subplot(155), plt.imshow(pred[b, 0, s1:e1, s2:e2], cmap='gray', vmax=1., vmin=0.)
            plt.show()

    # save_pesudo_label(pesudo_l, training_dataset_path + '/trainval/pixel_pseudo_label2.9', training_dataset_path + '/trainval/images')


def label_evolution(model_path, training_dataset_path):
    """
    进行标签进化的主函数。

    参数:
        model_path (str): 模型文件的路径。
        training_dataset_path (str): 训练数据集的路径。
    """
    # 加载模型
    ## cfg file
    with open("./cfg.yaml") as f:
        cfg = yaml.safe_load(f)
    # model = attenMultiplyUNet_withloss(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNANet_withloss(1, 
                    input_channels=3, 
                    block=Res_CBAM_block,
                    num_blocks=[2, 2, 2, 2],
                    nb_filter=[16, 32, 64, 128, 256],
                    deep_supervision=True,
                    grad_loss=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    # 加载数据集
    dataset = NUDTDataset(training_dataset_path, "train", 256, pt_label=True, pesudo_label=True, augment=False, turn_num=0.28, cfg=cfg)
    # dataset = IRSTD1kDataset(training_dataset_path, "train", 512, pt_label=True, pesudo_label=True, augment=False, turn_num='0.21', cfg=cfg)
    dataloader = Data.DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)

    pesudo_l = []
    img_idx = 0
    preds = []
    # 遍历数据集进行标签进化
    for i, (image, pt_label, pesudo_label) in enumerate(dataloader):
        # 预测
        with torch.no_grad(): 
            image = image.to(device)
            image_ = image.repeat(1, 3, 1, 1)  # for DNANet
            # image_ = TF.resize(image_, (256, 256), antialias=True)  # for IRSTD-1K Dataset
            # pred, _, _, _, _ = model.net(image)
            pred, _ = model.net(image_)
        pred = pred[-1]
        # pred = F.interpolate(pred, scale_factor=2, mode='bilinear', align_corners=True)
        preds.append(pred.cpu())

    for i, (image, pt_label, pesudo_label) in enumerate(dataloader):
        # 预测
        pred = preds[i]
        pred_ = (pred > 0.1).float()
        # 截出点标签的区域
        indices = torch.where(pt_label > 1e-4)

        pesudo_l = pesudo_l + [[] for j in range(image.shape[0])]
        # d1,d2 = 3, 8
        for b, _, c1, c2 in zip(*indices):
            s1, e1, s2, e2 = proper_region(pred_[b, 0] + pesudo_label[b, 0], c1, c2)
            
            region = image[b:b+1, :, s1:e1, s2:e2]
            target_ = gradient_expand_one_size(region)

            target_, () = target_adanptive_filtering_v2(target_, region, pred[b, 0, s1:e1, s2:e2])
    
            # 优化伪标签
            target = fusion_tm_dl(target_, pred_[b, 0, s1:e1, s2:e2], 0.8, 0.4, 1)

            # final_target_ = target_adanptive_filtering(target, region, pred[b, 0, s1:e1, s2:e2])
            target = mapping_4_crf(target, 0.01, 0.7, 0.1)
            final_target_ = apply_crf(target, region[0,0].numpy(), 10)
            final_target_ = torch.tensor(final_target_)

            # 审查，新的伪标签和上一轮伪标签的差距在一定范围内，若差距过大，则还是使用上一轮的伪标签
            final_target = examine_iou(final_target_, pesudo_label[b, 0, s1:e1, s2:e2], iou_treshold=0.01)
            # 保存结果
            img_idx = i * 16 + b
            # target_info = {"target": 0}
            target_info = {"target": final_target, "idx": (s1, e1, s2, e2)}
            pesudo_l[img_idx].append(target_info)
            # # 显示结果
            # plt.figure(figsize=(30, 6))
            # plt.subplot(151), plt.imshow(region[0,0], cmap='gray', vmax=1., vmin=0.)
            # plt.subplot(152), plt.imshow(target_, cmap='gray', vmax=1., vmin=0.)
            # plt.subplot(153), plt.imshow(target, cmap='gray', vmax=1., vmin=0.)
            # plt.subplot(154), plt.imshow(final_target, cmap='gray', vmax=1., vmin=0.)
            # plt.subplot(155), plt.imshow(pred[b, 0, s1:e1, s2:e2], cmap='gray', vmax=1., vmin=0.)
            # plt.show()

    save_pesudo_label(pesudo_l, training_dataset_path + '/trainval/pixel_pseudo_label1.24', training_dataset_path + '/trainval/images')


def label_evolution_V2(model_path, training_dataset_path):
    """
    进行标签进化的主函数。

    参数:
        model_path (str): 模型文件的路径。
        training_dataset_path (str): 训练数据集的路径。
    """
    # 加载模型
    ## cfg file
    with open("./cfg.yaml") as f:
        cfg = yaml.safe_load(f)
    # model = attenMultiplyUNet_withloss(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNANet_withloss(1, 
                    input_channels=3, 
                    block=Res_CBAM_block,
                    num_blocks=[2, 2, 2, 2],
                    nb_filter=[16, 32, 64, 128, 256],
                    deep_supervision=True,
                    grad_loss=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    # 加载数据集
    dataset = NUDTDataset(training_dataset_path, "train", 256, pt_label=True, pesudo_label=True, augment=False, turn_num=0.32, cfg=cfg)
    # dataset = IRSTD1kDataset(training_dataset_path, "train", 512, pt_label=True, pesudo_label=True, augment=False, turn_num='0.22', cfg=cfg)
    dataloader = Data.DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)

    pesudo_l = []
    img_idx = 0
    preds = []
    # 遍历数据集进行标签进化
    for i, (image, pt_label, pesudo_label) in enumerate(dataloader):
        # 预测
        with torch.no_grad(): 
            image = image.to(device)
            image_ = image.repeat(1, 3, 1, 1)  # for DNANet
            # image_ = TF.resize(image_, (256, 256), antialias=True)  # for IRSTD-1K Dataset
            # pred, _, _, _, _ = model.net(image)
            pred, _ = model.net(image_)
        pred = pred[-1]
        # pred = F.interpolate(pred, scale_factor=2, mode='bilinear', align_corners=True)
        preds.append(pred.cpu())

    for i, (image, pt_label, pesudo_label) in enumerate(dataloader):
        # 预测
        pred = preds[i]
        pred_ = (pred > 0.1).float()
        # 截出点标签的区域
        indices = torch.where(pt_label > 1e-4)

        pesudo_l = pesudo_l + [[] for j in range(image.shape[0])]
        # d1,d2 = 3, 8
        for b, _, c1, c2 in zip(*indices):
            s1, e1, s2, e2 = proper_region(pred_[b, 0] + pesudo_label[b, 0], c1, c2)
            
            region = image[b:b+1, :, s1:e1, s2:e2]
            target_ = gradient_expand_one_size(region)

            advice_region = examine_iou(pred_[b, 0, s1:e1, s2:e2] , pesudo_label[b, 0, s1:e1, s2:e2], iou_treshold=0.01)
            # 优化伪标签
            target_ = fusion_tm_dl(target_, advice_region, 0.8, 0.2)

            target, () = target_adanptive_filtering_v2(target_, region, advice_region)

            # final_target_ = target_adanptive_filtering(target, region, pred[b, 0, s1:e1, s2:e2])
            target = mapping_4_crf_v3(target, target_, 0.01, 0.75, 0.25)
            final_target_ = apply_crf(target, region[0,0].numpy(), 1)
            final_target_ = torch.tensor(final_target_)

            # 审查，新的伪标签和上一轮伪标签的差距在一定范围内，若差距过大，则还是使用上一轮的伪标签
            final_target = examine_iou(final_target_, pesudo_label[b, 0, s1:e1, s2:e2], iou_treshold=0.01)
            # 保存结果
            img_idx = i * 16 + b
            # target_info = {"target": 0}
            target_info = {"target": final_target, "idx": (s1, e1, s2, e2)}
            pesudo_l[img_idx].append(target_info)
            # # 显示结果
            # plt.figure(figsize=(30, 6))
            # plt.subplot(151), plt.imshow(region[0,0], cmap='gray', vmax=1., vmin=0.)
            # plt.subplot(152), plt.imshow(target_, cmap='gray', vmax=1., vmin=0.)
            # plt.subplot(153), plt.imshow(target, cmap='gray', vmax=1., vmin=0.)
            # plt.subplot(154), plt.imshow(final_target, cmap='gray', vmax=1., vmin=0.)
            # plt.subplot(155), plt.imshow(pred[b, 0, s1:e1, s2:e2], cmap='gray', vmax=1., vmin=0.)
            # plt.show()

    save_pesudo_label(pesudo_l, training_dataset_path + '/trainval/pixel_pseudo_label1.31', training_dataset_path + '/trainval/images')


if __name__ == "__main__":
    model_path = "result/20250806T00-11-27_ws3.6_0.22_nudt/best.pkl"
    training_dataset_path = "W:/DataSets/ISTD/NUDT-SIRST"
    # training_dataset_path = "W:/DataSets/ISTD/IRSTD-1k"
    label_evolution_V2(model_path, training_dataset_path)
