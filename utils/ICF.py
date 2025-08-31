import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from utils.sum_val_filter import min_pool2d
from utils.utils import compute_mask_pixel_distances_with_coords, extract_local_windows, min_positive_per_local_area, \
    compute_local_extremes, compute_weighted_mean_variance, random_select_from_prob_mask, select_complementary_pixels, \
    get_connected_mask_long_side, keep_negative_by_top3_magnitude_levels, add_uniform_points_cuda
from utils.adaptive_filter import filter_mask_by_points
from utils.refine import dilate_mask, erode_mask

from typing import Tuple, Optional, Dict, List


def initial_target(image: torch.Tensor, pt_label: torch.Tensor, threshold: float = 0.1):
    """
    Args:
        image: 输入图像 [H, W]
        threshold: 阈值，用于初步划分种子点是否为有效种子点，如前景种子点需>threshold，背景种子点需<threshold
    Returns:
        seed_cofidence: 概率图 [H, W, 2]
    """
    H, W = image.shape
    image = image.unsqueeze(0).unsqueeze(0)

    # 找局部最高值，再梯度强度➗局部最大值，进行局部归一化
    local_max = F.max_pool2d(image, (5,5), stride=1, padding=2)
    local_norm_image = image / (local_max + 1e-11)

    # 局部最大值
    local_max2 = F.max_pool2d(local_norm_image, (3,3), stride=1, padding=1)
    summit_mask = (local_norm_image == local_max2)
    summit_mask = summit_mask.squeeze(0).squeeze(0)
    image = image.squeeze(0).squeeze(0)
    fg_area = image > threshold
    bg_area = ~fg_area
    fg_seed = (summit_mask * fg_area).float()
    bg_seed = (summit_mask * bg_area).float()

    for iter1 in range(10):
        for iter2 in range(100):
            fig = plt.figure(figsize=(35, 5))
            plt.subplot(1, 7, 1)
            plt.imshow(image.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
            plt.subplot(1, 7, 2)
            plt.imshow(fg_seed.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
            plt.subplot(1, 7, 3)
            plt.imshow(bg_seed.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)

            fg_seed_num = np.ceil(fg_seed.sum().item())
            bg_seed_num = np.ceil(bg_seed.sum().item())

            if fg_seed_num > 3 and bg_seed_num > 3:

                # fg_vwo, fg_v = compute_weighted_mean_variance(image, fg_seed > 0.1, int(fg_seed_num/3))
                # bg_vwo, bg_v = compute_weighted_mean_variance(image, bg_seed > 0.1, int(bg_seed_num/3))
                local_fgseed_num = 3 if fg_seed_num // 3 < 3 else fg_seed_num // 3
                print('local_fgseed_num', local_fgseed_num)
                fg_mean, fg_var , _, _ = compute_weighted_mean_variance(image, fg_seed > 0.1, int(local_fgseed_num))
                local_bgseed_num = 6 if bg_seed_num // 3 < 6 else bg_seed_num // 3
                print('local_bgseed_num', local_bgseed_num)
                bg_mean, bg_var , _, _ = compute_weighted_mean_variance(image, bg_seed > 0.1, int(local_bgseed_num))

                fg_std = torch.abs(image - fg_mean) / (torch.sqrt(fg_var) + 1e-8)
                bg_std = torch.abs(image - bg_mean) / (torch.sqrt(bg_var) + 1e-8)

                result_ = fg_std - bg_std
                # result_ = fg_v/(fg_vwo+1e-8) -bg_v/(bg_vwo+1e-8)   #(H,W)

                print(local_fgseed_num, local_bgseed_num)
                print("result_")
                print(result_)
                print(fg_std)
                print(bg_std)
                # print((fg_v/(fg_vwo+1e-8) - 1))
                # print((bg_v/(bg_vwo+1e-8) - 1))

                plt.subplot(1, 7, 4)
                plt.imshow(result_, cmap='gray')
                # plt.subplot(1, 7, 5)
                # plt.imshow((fg_v/(fg_vwo+1e-8)-1), cmap='gray')
            else:
                result_ = -image + threshold

            fg_discuss_area = dilate_mask(fg_seed > 0.1, 1).bool() ^ (fg_seed > 0.1)
            masked_area = torch.where(fg_discuss_area & (result_ < 0.), result_, torch.tensor(float('inf')))
            k_actual = min((fg_discuss_area & (result_ < 0.)).float().sum(), fg_seed_num)
            fg_seed_new = torch.zeros_like(fg_seed)
            if k_actual > 0:
                val, idx = torch.topk(masked_area.view(-1), int(k_actual), largest=False)
                fg_seed_new = fg_seed_new.view(-1)
                fg_seed_new[idx] = 1.
                fg_seed_new = fg_seed_new.view(H, W)

            bg_discuss_area = dilate_mask(bg_seed > 0.1, 1).bool() ^ (bg_seed > 0.1)
            masked_area = torch.where(bg_discuss_area & (result_ > 0.), result_, torch.tensor(-float('inf')))
            k_actual = min((bg_discuss_area & (result_ > 0.)).float().sum(), bg_seed_num)
            bg_seed_new = torch.zeros_like(bg_seed)
            if k_actual > 0:
                val, idx = torch.topk(masked_area.view(-1), int(k_actual), largest=True)
                bg_seed_new = bg_seed_new.view(-1)
                bg_seed_new[idx] = 1.
                bg_seed_new = bg_seed_new.view(H, W)

            plt.subplot(1, 7, 6)
            plt.imshow(fg_seed_new, cmap='gray', vmax=1.0, vmin=0.)
            plt.subplot(1, 7, 7)
            plt.imshow(bg_seed_new, cmap='gray')
            plt.show(block=False)
            a = input()

            diff = torch.norm(fg_seed_new.float() + bg_seed_new.float()) 
            if diff < 2:
                # print(f"iter1 {iter1} iter2 Converged at {iter2}")
                break
            # else:
            #     print(f"iter1 {iter1} iter2 {iter2}, Diff: {diff}")
            
            decay_rate = 0.2
            fg_seed_, bg_seed_ = (fg_seed > 0.1) * (result_ < 0.), (bg_seed > 0.1) * (result_ >= 0.)
            fg_seed = fg_seed_new.float() + (1-decay_rate) * fg_seed + decay_rate * fg_seed_
            bg_seed = bg_seed_new.float() + (1-decay_rate) * bg_seed + decay_rate * bg_seed_
        
        # # 
        # fig = plt.figure(figsize=(15, 5))
        # plt.subplot(1, 3, 1)
        # plt.imshow(local_norm_image.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
        # plt.subplot(1, 3, 2)
        # plt.imshow(result, cmap='gray', vmax=1.0, vmin=0.0)
        # plt.subplot(1, 3, 3)
        # plt.imshow(fg_area, cmap='gray', vmax=1.0, vmin=0.0)
        # plt.show(block=False)
        # a = input()
        # 修改area
        fg_area_new = result_ < 0.
        bg_area_new = result_ > 0.
        diff = torch.norm(bg_area_new.float() - bg_area.float())
        if diff <= 1:
            # print(f"iter1 Converged at{iter1}")
            break
        # else:
        #     print(f"iter1 {iter1}, Diff: {diff}")
        fg_area, bg_area = fg_area_new, bg_area_new

    return fg_area