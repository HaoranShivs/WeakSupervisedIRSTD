import torch
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np

from scipy import ndimage
from scipy.ndimage import label as ndlabel
from scipy.signal import find_peaks as fpk

from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import time
from multiprocessing import shared_memory
import os
import os.path as osp
import yaml
import struct

from utils.refine import dilate_mask, erode_mask
from utils.utils import compute_weighted_centroids, mask_diameter, farthest_point_sampling, iou_score
from utils.RandomWalk import process_image_4_RW, RandomWalkPixelLabeling
from utils.grad_expand_utils import *
from utils.adaptive_filter import *
from utils.label_evolution_utils import *

from net.dnanet import DNANet_withloss, Res_CBAM_block
from data.sirst import IRSTD1kDataset, NUDTDataset
from net.attentionnet import attenMultiplyUNet_withloss

# # 设置pytorch打印选项
# torch.set_# printoptions(
#     threshold=1024,  # 最大显示元素数量为10
#     linewidth=320,  # 每行的最大字符数为120
#     precision=20,  # 小数点后的数字精度为4
#     edgeitems=20,  # 每行显示的边缘元素数量为5
#     sci_mode=False,  # 不使用科学计数法
# )

cfg_path = "cfg.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description="Implement of BaseNet")

    #
    # Configuration
    #
    parser.add_argument("--cfg-path", type=str, default="./cfg.yaml", help="path of cfg file")

    parser.add_argument("--dataset", type=str, default="nudt", help="choose datasets")

    parser.add_argument("--save-folder", type=str, default="pixel_pseudo_label0", help="folder name of pseudo label directory")

    parser.add_argument("--debug", type=bool, default=0, help="whether to use chart")

    parser.add_argument("--model-path", type=str, default="", help="path of DLmodel")

    parser.add_argument("--last-turnnum", type=str, default="0", help="Last turn_num of pseudo label")

    args = parser.parse_args()

    return args


def gradient_expand_one_size(region, scale_weight=[0.5, 0.5, 0.5], view=False):
    # 归一化区域像素，以形成更强烈的边缘
    region_ = (region - region.min())/(region.max() - region.min())
    # 梯度形成
    img_gradient_1 = img_gradient2(region_)  # 2*2 sober
    img_gradient_2 = img_gradient3(region_)  # 3*3 sober
    img_gradient_3 = img_gradient5(region_)  # 5*5 sobel
    
    # 梯度映射，突出中灰度的区分度
    img_gradient_1, img_gradient_2, img_gradient_3 = sigmoid_mapping3(img_gradient_1, 10), \
         sigmoid_mapping3(img_gradient_2, 10), sigmoid_mapping3(img_gradient_3, 10)

    # 多尺度融合
    img_gradient_ = grad_multi_scale_fusion(img_gradient_1, scale_weight[0]) * grad_multi_scale_fusion(img_gradient_2, scale_weight[0]) * \
        grad_multi_scale_fusion(img_gradient_3, scale_weight[0])
    img_gradient_ = (img_gradient_ - img_gradient_.min())/(img_gradient_.max() - img_gradient_.min() + 1e-22)
    # 用单像素宽度的梯度替代模糊边缘的宽的梯度
    grad_mask = local_max_gradient(img_gradient_)
    img_gradient_4 = grad_mask * img_gradient_

    # 扩展梯度
    grad_boundary = boundary4gradient_expand(img_gradient_4, 1e20)
    expanded_grad = img_gradient_4
    region_size = region.shape[2] if region.shape[2] > region.shape[3] else region.shape[3]
    for z in range(region_size):
        expanded_grad_ = gradient_expand_one_step(expanded_grad)
        expanded_grad_ += grad_boundary
        expanded_grad_ = torch.where(expanded_grad > expanded_grad_, expanded_grad, expanded_grad_) * (expanded_grad_ > -1e-4)
        expanded_grad = expanded_grad_

    _target = torch.sum(expanded_grad[0], dim=0)
    _target = (_target - _target.min())/(_target.max() - _target.min())

    # # 显示结果
    # angle_idx = 3
    # grad_boundary_4show = torch.cat((-grad_boundary[:,angle_idx]*1e-20, torch.zeros_like(grad_boundary[:,angle_idx]), torch.zeros_like(grad_boundary[:,angle_idx])), dim=0)
    # plt.figure(figsize=(25, 5))
    # plt.subplot(151), plt.imshow(region[:,0].repeat(3,1,1).permute(1,2,0)), plt.axis('off')
    # plt.subplot(152), plt.imshow((img_gradient_4[:,angle_idx].repeat(3,1,1) + grad_boundary_4show).permute(1,2,0)), plt.axis('off')
    # plt.subplot(153), plt.imshow(grad_boundary_4show.permute(1,2,0)), plt.axis('off')
    # plt.subplot(154), plt.imshow(expanded_grad[:,angle_idx].repeat(3,1,1).permute(1,2,0)), plt.axis('off')
    # plt.subplot(155), plt.imshow(_target.unsqueeze(0).repeat(3,1,1).permute(1,2,0)), plt.axis('off')
    # plt.show()
    if view:
        return _target, (img_gradient_1, img_gradient_2, img_gradient_3, img_gradient_4, grad_mask, grad_boundary, expanded_grad)
    return _target, ()


def target_adanptive_filtering(target, local_method=0, view=False):
    # 归一化到 0-255 范围内
    min_val = target.min()
    max_val = target.max()
    target_normal = ((target - min_val) / (max_val - min_val) * 255).type(torch.uint8)
    # 过滤梯度扩展的图形结果
    hist, bins = compute_histogram(target_normal)
    # #统计梯度
    # global global_hist
    # global hist_cnt
    # # global_hist = target_adanptive_filtering_v2(img_gradient_1.squeeze(0).squeeze(0), global_hist)
    # global_hist = hist + global_hist
    # hist_cnt += 1
    ## 为了降低低值像素的巨大数量对平滑曲线的巨大影响，我们直接限制最大值为5
    # limitation = 3  # nudt是5
    # hist = torch.where(hist > limitation, torch.ones_like(hist) * limitation, hist)
    hist = hist_mapping(hist, 0.5)
    ## 平滑处理
    smooth_hist = smooth_histogram(hist.numpy(), 3, 3)
    smoother_hist = smooth_histogram(hist.numpy(), 10, 3)    # 更加全局的曲线

    # 在 smoother_hist 中找出所有波谷
    peaks, props = fpk(-smooth_hist, prominence=0.01, width=3, distance=1)
    peaks_2, props_2 = fpk(-smoother_hist, prominence=0.001, width=3, distance=1)
    
    def proper_peak2(peaks, props, peaks2, props2, pred=None, target=None):
        """
        根据波峰的属性，选出range范围内最合适的波峰。
        属性包括：显著性，波峰宽度，靠经第一个波峰（小比例），靠近大波峰, 所靠近的大波峰的宽度，显著性
        参数：
            peaks (list): 波峰的索引列表。
            props (dict): 波峰的属性字典，包括 prominence、left_bases 和 right_bases 等。
            peaks2 (list): 大波峰的索引列表。
            props2 (dict): 大波峰的属性字典，包括 prominence、left_bases 和 right_bases 等。
            pred: numpy.array, 预测的灰度值。
        返回：
            int: 最合适的波峰的索引。
        注意：
            如果波峰的数量为0，则返回-1。
        """
        if len(peaks) == 0:
            return 0
        score = [ 0 for i in range(len(peaks))]

        # 波峰宽度
        ratio = 1.0
        width_rank = np.argsort(props['widths'])
        width_score = []
        for i in range(len(score)):
            width_score.append(props['widths'][i]/props['widths'][width_rank[-1]])
            score[i] = score[i] + width_score[i] * ratio
        # print(width_score)

        # 显著性
        ratio = 1.0
        prominence_rank = np.argsort(props['prominences'])
        prominence_score = []
        for i in range(len(score)):
            prominence_score.append(props['prominences'][i]/props['prominences'][prominence_rank[-1]])
            score[i] = score[i] + prominence_score[i] * ratio
        # print(prominence_score)
        
        # 边缘部分清洁性
        close_scores = []
        for i in range(len(peaks)):
            target_ = (target > peaks[i]).type(torch.float32)
            close_score = object_closed_score(target_, 8)
            close_scores.append(close_score)
        max_close_score = np.max(close_scores)
        ratio = 2.0
        for i in range(len(score)):
            close_scores[i] = close_scores[i]/max_close_score
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
            target_ = (target > peaks[i]).type(torch.float32)
            integirty_score = calculate_spatial_discontinuity(target_)
            integirty_scores.append(integirty_score)
        # print(integirty_scores)
        max_integirty_score = np.max(integirty_scores)
        ratio = 4.0
        for i in range(len(score)):
            integirty_scores[i] = 1 - integirty_scores[i]/max_integirty_score
            score[i] = score[i] + integirty_scores[i] * ratio
        # print(integirty_scores)

        # 高灰度值保留性
        highval_keeping_scores = []
        for i in range(len(peaks)):
            target_ = (target > peaks[i]) * target
            highval_keeping_score = target_.sum()
            highval_keeping_scores.append(highval_keeping_score)
        max_highval_keeping_score = np.max(highval_keeping_scores)
        ratio = 2.0
        for i in range(len(score)):
            highval_keeping_scores[i] = highval_keeping_scores[i]/max_highval_keeping_score
            score[i] = score[i] + highval_keeping_scores[i] * ratio
        # print(highval_keeping_scores)

        # print('score', score)
        peaks_idx = np.argmax(score)
        return peaks_idx, score

    threshold = 0.
    if len(peaks) > 0:
        peak_idx, peak_score = proper_peak2(peaks, props, peaks_2, props_2, target=target_normal)
        threshold = peaks[peak_idx]

    filtered_target = target_normal * (target_normal > threshold)

    # # 绘制直方图
    # fig = plt.figure(figsize=(15, 5))
    
    # # 原始图像
    # plt.subplot(1, 3, 1)
    # plt.imshow(target, cmap='gray')
    
    # # 直方图
    # plt.subplot(1, 3, 2)
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

    if view:
        return filtered_target, (bins, hist, smooth_hist, smoother_hist, peaks, peaks_2, threshold)
    return filtered_target, peak_score


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
            pred: numpy.array, 预测的灰度值。
            target: numpy.array, 由传统算法预测的logits
        返回：
            peaks_prob (list): peaks作为分类阈值得概率
        注意：
            如果波峰的数量为0，则返回-1。
        """
        if len(peaks) == 0:
            return -1
        score = [ 0 for i in range(len(peaks))]
        
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
        # close_scores = mapping_list(close_scores, 0.01)
        max_close_score = np.max(close_scores) 
        for i in range(len(close_scores)):
            close_scores[i] = close_scores[i]/max_close_score
        ratio = 2.0
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
        ratio = 4.0
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
        ratio = 2.0
        for i in range(len(score)):
            score[i] = score[i] + highval_keeping_scores[i] * ratio

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
        # peak_idx = []
        # for i in sorted_score_idx[-2:]:
        #     if i <= sorted_score_idx[-1]:
        #         prob = (score[i] - score[sorted_score_idx[0]]) / (score[sorted_score_idx[-1]] - score[sorted_score_idx[0]])
        #     else:
        #         prob = 1 + (score[sorted_score_idx[-1]] - score[i]) / (score[sorted_score_idx[-1]] - score[sorted_score_idx[0]])
        #     probs.append(prob)
        #     peak_idx.append(i)
            # peak_idx.append(i)
        return probs

    peak_probs = peaks_probability(peaks, props, None, target_normal.numpy())

    filtered_target = torch.zeros_like(target_normal)
    for i in range(len(peaks)):
        threshold_filtered_area = (target_normal > peaks[i]).float() * peak_probs[i]
        filtered_target = torch.where(threshold_filtered_area > filtered_target, threshold_filtered_area, filtered_target)
        # filtered_target = torch.where(threshold_filtered_area > 0., threshold_filtered_area, filtered_target)
        # print(peaks[i], peak_probs[i])
    
    threshold = 0.

    # 绘制直方图
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
    # plt.imshow(filtered_target, cmap='gray', vmax=1.0, vmin=0.0)
    # plt.title('Filtered Image')

    # plt.show()

    if view:
        return filtered_target, (bins, hist, smooth_hist, smoother_hist, peaks, peaks_2, threshold)
    return filtered_target, ()


def gradient_expand_filter(img, pt_label, region_size, view=False):
    B, _, H, W = img.shape
    indices = torch.where(pt_label > 1e-4)
    output = torch.zeros_like(img, dtype=torch.float32)
    for b, _, s1, s2 in zip(*indices):
        # 提取区域
        targets = []
        for i in range(len(region_size)):
            _region_size = region_size[i]
            # 计算区域坐标（y1, x1)(y2, x2)
            y1 = max(1, s1 - _region_size // 2)
            x1 = max(1, s2 - _region_size // 2)
            y2 = min(H-1, s1 + (_region_size - _region_size // 2))
            x2 = min(W-1, s2 + (_region_size - _region_size // 2))
            _region = img[b:b+1,:1, y1:y2, x1:x2]

            _target, grad_expand_process_data = gradient_expand_one_size(_region, view)
            targets.append({'target':_target, 'coor':(y1,x1,y2,x2)})
        
        final_target, scores, coors = finalize_target(targets, view)
        target_filtered, treshold_filter_process_data = target_adanptive_filtering_v2(final_target, img[b,0, coors[0]:coors[2], coors[1]:coors[3]], view=view)

        target_filtered_ = mapping_4_crf(target_filtered, final_target, 0.01, 0.60, 0.40)
        target_filtered__ = dense_crf(target_filtered_.numpy(), img[b,0, coors[0]:coors[2], coors[1]:coors[3]].numpy(), 10)
        target_filtered__ = torch.tensor(target_filtered__)

        target_filtered_by_points = filter_mask_by_points(target_filtered, pt_label[b,0, coors[0]:coors[2], coors[1]:coors[3]]) # (uint8)
        target_refined = target_filtered_by_points
        output[b,0, coors[0]:coors[2], coors[1]:coors[3]] = torch.max(output[b,0, coors[0]:coors[2], coors[1]:coors[3]], target_refined)
        if view:
            process_data_view(img[b], _region[0,0], _target, grad_expand_process_data, final_target, scores, target_filtered, treshold_filter_process_data, target_refined, output[b,0])
    return output


def label_evolution(image, pt_label, pesudo_label, pred, view=False):
    pred_ = (pred > 0.1).float()
    # 截出点标签的区域
    indices = torch.where(pt_label > 1e-4)
    output = torch.zeros_like(image, dtype=torch.float32)
    # d1,d2 = 3, 8
    for b, _, c1, c2 in zip(*indices):
        s1, e1, s2, e2 = proper_region(pred_[b, 0] + pesudo_label[b, 0], c1, c2)
        
        region = image[b:b+1, :, s1:e1, s2:e2]
        target_, grad_expand_process_data = gradient_expand_one_size(region, [0.75, 0.75, 0.25], view=view)

        advice_region = examine_iou(pred_[b, 0, s1:e1, s2:e2] , pesudo_label[b, 0, s1:e1, s2:e2], iou_treshold=0.01)
        # 优化伪标签
        target_fused = fusion_tm_dl(target_, advice_region, 0.7, 0.3)

        target, treshold_filter_process_data = target_adanptive_filtering_v2(target_fused, region, advice_region, view=view)

        # final_target_ = target_adanptive_filtering(target, region, pred[b, 0, s1:e1, s2:e2])
        target_mapped = mapping_4_crf_v3(target, target_fused, 0.01, 0.75, 0.25)
        final_target_ = dense_crf(target_mapped, region[0,0].numpy(), 1)
        final_target_ = torch.tensor(final_target_)

        # 审查，新的伪标签和上一轮伪标签的差距在一定范围内，若差距过大，则还是使用上一轮的伪标签
        final_target = examine_iou(final_target_, pesudo_label[b, 0, s1:e1, s2:e2], iou_treshold=0.01)
        # 保存结果
        output[b,0, s1:e1, s2:e2] = torch.max(output[b,0, s1:e1, s2:e2], final_target)
        # # 显示结果
        # plt.figure(figsize=(30, 6))
        # plt.subplot(151), plt.imshow(region[0,0], cmap='gray', vmax=1., vmin=0.)
        # plt.subplot(152), plt.imshow(target_, cmap='gray', vmax=1., vmin=0.)
        # plt.subplot(153), plt.imshow(target, cmap='gray', vmax=1., vmin=0.)
        # plt.subplot(154), plt.imshow(final_target, cmap='gray', vmax=1., vmin=0.)
        # plt.subplot(155), plt.imshow(pred[b, 0, s1:e1, s2:e2], cmap='gray', vmax=1., vmin=0.)
        # plt.show()
        if view:
            process_data_view(image[b], region[0,0], pred[b, 0, s1:e1, s2:e2], grad_expand_process_data, target_fused, [100,], target, treshold_filter_process_data, final_target, output[b,0])
    return output


def process_data_view(img, region, target, gred_expand_process_data, final_target, scores, target_filtered, treshold_filter_process_data, target_refined, mask_filtered_result):
    def contrast_view(image, mask_pred, mask_gt=None):
        # 如果 image 是归一化的 [0, 1]，则反归一化到 [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # 将图像从 CxHxW 转换为 HxWxC（如果是的话）
        if len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))

        # 创建彩色掩膜
        def apply_mask(image, image_color, mask, mask_color, alpha=0.5):
            if image.shape[2] == 1:
                image = np.concatenate([image, image, image], axis=2)
            for c in range(3):
                image[:, :, c] = image[: ,: , c] if image_color[c] == 1 else np.zeros_like(image[:, :, c])
                image[:, :, c] = np.where(mask > 0.1,
                                        image[:, :, c] * (1 - alpha) + alpha * mask_color[c] * 255,
                                        image[:, :, c])
            return image

        # 定义颜色（RGB格式，范围 0~1）
        color_gt = [0, 1, 0]    # 绿色表示 ground truth
        color_pred = [1, 0, 0]  # 红色表示 prediction
        color_image = [1, 1, 1]

        image_pred = apply_mask(image.copy(), color_image, mask_pred, color_pred)
        if mask_gt != None:
            gt_pred = apply_mask(mask_gt.copy(), color_gt, mask_pred, color_pred)
            return gt_pred , image_pred
        return image_pred
    # 创建一个 3x4 的子图网格
    fig, axes = plt.subplots(4, 4, figsize=(20, 10))  # 调整 figsize 以适应布局
    
    ax = axes[0, 0]
    ax.imshow(img[0], cmap='gray')
    ax.set_title("image")

    ax = axes[0, 1]
    ax.imshow(region, cmap='gray')
    ax.set_title("target_region")
    
    ax = axes[0, 2]
    ax.imshow(mask_filtered_result, cmap='gray')
    ax.set_title("final_result")

    img_pred = contrast_view(img.numpy(), mask_filtered_result.numpy())
    ax = axes[0, 3]
    ax.imshow(img_pred)
    ax.set_title("contrast")

    grad1, grad2, grad3, final_grad, mask, boundary, expanded_grad = gred_expand_process_data

    ax = axes[1, 0]
    ax.imshow(torch.sum(final_grad, dim=[0,1]), cmap='gray')
    ax.set_title("final_gradient")

    ax = axes[1, 1]
    ax.imshow(target, cmap='gray')
    ax.set_title("expanded_gradient")

    ax = axes[1, 2]
    ax.imshow(torch.sum(grad1, dim=[0,1]), cmap='gray')
    ax.set_title("grad1")
    
    ax = axes[1, 3]
    ax.imshow(torch.sum(grad2, dim=[0,1]), cmap='gray')
    ax.set_title("grad2")

    ax = axes[2, 3]
    ax.imshow(torch.sum(grad3, dim=[0,1]), cmap='gray')
    ax.set_title("grad3")

    ax = axes[2, 2]
    ax.imshow(target_filtered, cmap='gray')
    ax.set_title("target_filtered by treshold")

    ax = axes[2, 0]
    ax.imshow(final_target, cmap='gray')
    ax.set_title(f"cofidence is respectively {scores}")

    bins, hist, smoothed_hist, smoothed_hist_2, valley_idx, valley_idx_2, threshold = treshold_filter_process_data

    # 直方图
    ax = axes[2, 1]
    ax.bar(bins, hist, color='blue', alpha=0.7, label='Histogram')
    ax.plot(bins, smoothed_hist, color='orange', label='Smoothed Histogram')
    ax.plot(bins, smoothed_hist_2, color='green', label='Smoothed Histogram_2')
    if valley_idx is not None:
        for i in valley_idx:
            ax.axvline(x=i, color='red', linestyle='--')
    if valley_idx_2 is not None:
        for i in valley_idx_2:
            ax.axvline(x=i, color='cyan', linestyle='--')
    ax.axvline(x=threshold, color='purple', linestyle='--') 
    ax.legend()
    ax.set_title('Brightness Histogram')
    ax.set_xlabel('Brightness Level')
    ax.set_ylabel('Pixel Count')
    ax.set_ylim(0, 2)  # 设置bottom和top为你想要的y轴范围

    ax = axes[3, 0]
    ax.imshow(target_refined, cmap='gray')
    ax.set_title("target_refined by local_contrast")

    ax = axes[3, 1]
    ax.imshow(boundary[0,0], cmap='gray')
    ax.set_title("boundary")

    ax = axes[3, 2]
    ax.imshow(boundary[0,6], cmap='gray')
    ax.set_title("boundary")

    ax = axes[3, 3]
    ax.imshow(boundary[0,3], cmap='gray')
    ax.set_title("boundary")


    # 调整子图间距
    plt.tight_layout()
    plt.show()
    a = input()
    # fig.canvas.draw()  # 绘制图像
    # image_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    # image_shape = fig.canvas.get_width_height()[::-1] + (4,)  # 获取图像形状 (height, width, channels)
    # plt.close(fig)  # 关闭图像以释放资源
    # save_plot_to_shared_memory(image_data, image_shape)
    # # print(f"Generated Chart gradient_expand")
 

def save_pesudo_label(pseudo, save_path, names):
    """
    将伪标签保存到指定路径。

    参数:
        pseudo (torch.tensor): 伪标签列表。(N, 1, H, W)
        save_path (str): 保存路径。
        names (str): 标签名称。
    """
    for i in range(pseudo.shape[0]):
        pesudo_label = pseudo[i,0].cpu().numpy()
        # 归一化到 0-255 范围内
        min_val = pesudo_label.min()
        max_val = pesudo_label.max()
        if max_val - min_val == 0:
            pesudo_label = np.zeros_like(pesudo_label, dtype=np.uint8)
        else:
            pesudo_label = ((pesudo_label - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        # 保存伪标签
        pesudo_label = Image.fromarray(pesudo_label, mode='L')  # 'L' 表示灰度模式
        pesudo_label.save(save_path + '/' + names[i])


def main(args):
    ## cfg file
    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    file_name = ''
    if args.debug == True:
        file_name = '_hard'
    # dataset
    if args.dataset == "nudt":
        trainset = NUDTDataset(base_dir=r"W:/DataSets/ISTD/NUDT-SIRST", mode="train", base_size=256, pt_label=True, \
                               pesudo_label=True, augment=False, turn_num=args.last_turnnum, file_name=file_name, cfg=cfg)
        img_path = "W:/DataSets/ISTD/NUDT-SIRST/trainval/images" 
    # elif args.dataset == 'sirstaug':
    #     trainset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
    #                                mode='train', base_size=args.base_size)  # base_dir=r'E:\ztf\datasets\sirst_aug'
    elif args.dataset == "irstd1k":
        trainset = IRSTD1kDataset(base_dir=r"W:/DataSets/ISTD/IRSTD-1k", mode="train", base_size=512, pt_label=True, \
                                  pesudo_label=True, augment=False, turn_num=args.last_turnnum, file_name=file_name, cfg=cfg)
        img_path = "W:/DataSets/ISTD/IRSTD-1k/trainval/images"
    else:
        raise NotImplementedError
    
    train_data_loader = Data.DataLoader(trainset, batch_size=32, shuffle=False, drop_last=False)

    # DLmodel
    model = DNANet_withloss(1, 
                            input_channels=3, 
                            block=Res_CBAM_block,
                            num_blocks=[2, 2, 2, 2],
                            nb_filter=[16, 32, 64, 128, 256],
                            deep_supervision=True,
                            grad_loss=False)
    if args.model_path != "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(args.model_path + '/best.pkl'))
        model.eval()
        model = model.to(device)

    names = os.listdir(img_path)
    save_path = img_path + '/../' + 'pixel_pseudo_label' + f'{args.save_folder}'

    # Save folders 
    if not os.path.exists(save_path):
        os.makedirs(save_path) 

    for j, (img, pt_label, pesudo_label) in enumerate(train_data_loader):
        if args.model_path != "":
            # 预测
            with torch.no_grad(): 
                image_ = img.to(device)
                image_ = image_.repeat(1, 3, 1, 1)  # for DNANet
                # image_ = TF.resize(image_, (256, 256), antialias=True)  # for IRSTD-1K Dataset
                # pred, _, _, _, _ = model.net(image)
                pred, _ = model.net(image_)
            pred = pred[-1]
            pred = pred.cpu()
            # pred = F.interpolate(pred, scale_factor=2, mode='bilinear', align_corners=True)

            target_grad_expanded_filtered = label_evolution(img, pt_label, pesudo_label, pred, view=args.debug)
        else:
            target_grad_expanded_filtered = gradient_expand_filter(img, pt_label, [8,16,32,48], view=args.debug)
        save_pesudo_label(target_grad_expanded_filtered, save_path, names[j*32: j*32+img.shape[0]])
        


# global_hist = torch.zeros((256,), dtype=torch.int32)
# hist_cnt = 0

if __name__ == "__main__":
    # 将图像数据写入共享内存
    args = parse_args()
    main(args)

    # print(global_hist, hist_cnt)

    # global_hist = global_hist.float() / hist_cnt
    # global_hist = hist_mapping(global_hist, 0.5)
    # bins = torch.arange(256)

    # # 绘制直方图
    # fig = plt.figure(figsize=(5, 5))
    # # 直方图
    # plt.plot(bins, global_hist, color='orange', label='Smoothed Histogram')

    # plt.legend()
    # # plt.title('Brightness curve')
    # plt.xlabel('Brightness Level')
    # plt.ylabel('Pixel Count: 1-e^(-0.5*y)')
    # plt.ylim(0, 2)  # 设置bottom和top为你想要的y轴范围
    
    # # # 过滤后的图像
    # # plt.subplot(1, 3, 3)
    # # plt.imshow(filtered_target, cmap='gray', vmax=1.0, vmin=0.0)
    # # plt.title('Filtered Image')

    # plt.show()
