import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from utils.sum_val_filter import min_pool2d
from utils.utils import compute_mask_pixel_distances_with_coords, extract_local_windows, min_positive_per_local_area, \
    compute_local_extremes, compute_weighted_mean_variance, random_select_from_prob_mask, select_complementary_pixels, \
    get_connected_mask_long_side, keep_negative_by_top3_magnitude_levels, add_uniform_points_cuda, big_num_mask, add_uniform_points_v2, \
    add_uniform_points_v3, get_min_value_outermost_mask, periodic_function, smooth_optim, bilateral_smooth_logits
from utils.adaptive_filter import filter_mask_by_points
from utils.refine import dilate_mask, erode_mask

from typing import Tuple, Optional, Dict, List


def initial_target(grad_intensity: torch.Tensor, pt_label: torch.Tensor, threshold: float = 0.1, fg_area=None, bg_area=None):
    """
    Args:
        grad_intensity: 输入图像 [H, W]
        threshold: 阈值，用于初步划分种子点是否为有效种子点，如前景种子点需>threshold，背景种子点需<threshold
    Returns:
        seed_cofidence: 概率图 [H, W, 2]
    """
    H, W = grad_intensity.shape

    # 找局部最高值，再梯度强度➗局部最大值，进行局部归一化
    local_max = F.max_pool2d(grad_intensity.unsqueeze(0).unsqueeze(0), (5,5), stride=1, padding=2)
    local_norm_GI = grad_intensity / (local_max + 1e-11)

    # 局部最大值
    local_max2 = F.max_pool2d(local_norm_GI, (3,3), stride=1, padding=1)
    summit_mask = (local_norm_GI == local_max2)
    local_norm_GI = local_norm_GI.squeeze(0).squeeze(0)
    summit_mask = summit_mask.squeeze(0).squeeze(0)

    fg_area = torch.min((grad_intensity > 0.5).float(), fg_area) if fg_area is not None else (grad_intensity > 0.5).float()
    if fg_area.sum() < 9:
        fg_area = (grad_intensity >=grad_intensity.view(-1).sort(descending=True).values[9]).float()
    bg_area = torch.max((grad_intensity < threshold).float(), bg_area) if bg_area is not None else (grad_intensity < threshold).float()
    if bg_area.sum() < 9:
        bg_area = (grad_intensity <= grad_intensity.view(-1).sort(descending=True).values[-9]).float()

    noise_ratio = 0.05
    converge_time = 0
    for iter1 in range(50):
        fg_mask = (summit_mask * (fg_area > 0.1)).float()
        bg_mask = (summit_mask * (bg_area > 0.1)).float()
        for iter2 in range(100):
            noise = torch.rand(grad_intensity.shape)
            GI = grad_intensity * (1-noise_ratio) + noise * noise_ratio
            # fig = plt.figure(figsize=(35, 5))
            # plt.subplot(1, 7, 1)
            # plt.imshow(GI.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
            # plt.subplot(1, 7, 2)
            # plt.imshow(fg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
            # plt.subplot(1, 7, 3)
            # plt.imshow(bg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)

            max_num = np.ceil(fg_mask.sum().item())
            min_num = np.ceil(bg_mask.sum().item())

            if max_num > 8 and min_num > 8:
                fg_ratio = 1 - max_num /(fg_area.sum() + 1e-8)
                bg_ratio = 1 - min_num /(bg_area.sum() + 1e-8)
                fg_ratio = max(fg_ratio, 0.20)
                bg_ratio = max(bg_ratio, 0.20)

                local_max_num = 9 if max_num * fg_ratio < 9 else max_num * fg_ratio
                _, fg_vwo, _, fg_v = compute_weighted_mean_variance(GI, fg_mask > 0.1, int(local_max_num))

                local_min_num = 9 if min_num * bg_ratio < 9 else min_num * bg_ratio
                _, bg_vwo, _, bg_v = compute_weighted_mean_variance(GI, bg_mask > 0.1, int(local_min_num))
                # print(local_max_num, local_min_num)
                result_ = fg_v/(fg_vwo+1e-8) - bg_v/(bg_vwo+1e-8)   #(H,W)
                # result_ = keep_negative_by_top2_magnitude_levels(result_)

                # print(GI)
                # print("result_")
                # print(result_)
                # print(fg_vwo)
                # print(bg_vwo)
                # print((fg_v/(fg_vwo+1e-8) - 1))
                # print((bg_v/(bg_vwo+1e-8) - 1))

                # plt.subplot(1, 7, 4)
                # plt.imshow((bg_v/(bg_vwo+1e-8)-1), cmap='gray')
                # plt.subplot(1, 7, 5)
                # plt.imshow((fg_v/(fg_vwo+1e-8)-1), cmap='gray')
            else:
                result_ = torch.where(fg_area > 0.1, -GI, GI)
            result_ = keep_negative_by_top3_magnitude_levels(result_, target_size=fg_area.sum())
            result = torch.where(result_ < 0., torch.ones_like(GI), torch.zeros_like(GI)).bool()
            # result = filter_mask_by_points(result, pt_label, kernel_size=5).bool()

            # plt.subplot(1, 7, 6)
            # plt.imshow(result, cmap='gray', vmax=1.0, vmin=0.)
            # plt.subplot(1, 7, 7)
            # plt.imshow(result_, cmap='gray')
            # plt.show(block=False)
            # a = input()

            fg_seed_num = int(0.1*max_num)
            fg_seed_num = fg_seed_num if fg_seed_num > 2 else 2
            # if fitted != 1:
            fg_mask_new = add_uniform_points_v3(GI, (fg_area > 0.1) * (result_ < 0.), fg_mask>0.1, int(fg_seed_num), mode='fg')
            # else:
            # fg_mask_new = add_uniform_points_cuda((fg_area > 0.1) * (result_ < 0.), fg_mask>0.1, int(fg_seed_num))
            fg_mask_new = fg_mask_new.bool() * result

            bg_seed_num = int(0.1*min_num)
            bg_seed_num = bg_seed_num if bg_seed_num > 8 else 8
            # if fitted != 1:
            # bg_mask_new = add_uniform_points_cuda((bg_area > 0.1) * (result_ >= 0.), bg_mask>0.1, int(bg_seed_num))
            bg_mask_new = add_uniform_points_v3(GI, (bg_area > 0.1) * (result_ >= 0.), bg_mask>0.1, int(bg_seed_num), mode='bg')
            # else:
            # bg_mask_new = add_uniform_points_cuda((bg_area > 0.1) * (result_ > 0.), bg_mask>0.1, int(bg_seed_num))
            bg_mask_new = bg_mask_new.bool() * ~result

            diff = torch.norm(fg_mask_new.float() - (fg_mask > 0.1).float()) 
            if diff < 1:
                # print(f"iter1 {iter1} iter2 Converged at {iter2}")
                break
            # else:
            #     print(f"iter1 {iter1} iter2 {iter2}, Diff: {diff}")

            decay_rate = 0.8
            fg_mask, bg_mask = fg_mask_new.float() + fg_mask*decay_rate, bg_mask_new.float() + bg_mask*decay_rate
            fg_mask, bg_mask = torch.clamp(fg_mask, min=0.0, max=1.0), torch.clamp(bg_mask, min=0.0, max=1.0)
        
        result_total = torch.zeros_like(result_)
        for i in range(10):
            noise = torch.rand(GI.shape)
            GI = grad_intensity * (1-noise_ratio) + noise * noise_ratio

            local_max_num = 9 if max_num * fg_ratio < 9 else max_num * fg_ratio
            _, fg_vwo, _, fg_v = compute_weighted_mean_variance(GI, fg_mask > 0.1, int(local_max_num))

            local_min_num = 9 if min_num * bg_ratio < 9 else min_num * bg_ratio
            _, bg_vwo, _, bg_v = compute_weighted_mean_variance(GI, bg_mask > 0.1, int(local_min_num))
            result_total += fg_v/(fg_vwo+1e-8) -bg_v/(bg_vwo+1e-8)   #(H,W)
        result_total = keep_negative_by_top3_magnitude_levels(result_total, target_size=fg_area.sum())
        result = torch.where(result_total < 0., torch.ones_like(GI), torch.zeros_like(GI)).bool()

        # fig = plt.figure(figsize=(25, 5))
        # plt.subplot(1, 5, 1)
        # plt.imshow(GI.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
        # plt.subplot(1, 5, 2)
        # plt.imshow(result, cmap='gray', vmax=1.0, vmin=0.0)
        # plt.subplot(1, 5, 3)
        # plt.imshow(bg_area, cmap='gray', vmax=1.0, vmin=0.0)
        # plt.show(block=False)
        # plt.subplot(1, 5, 4)
        # plt.imshow(fg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
        # plt.subplot(1, 5, 5)
        # plt.imshow(bg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
        # a = input()
        # 修改area
        # result = filter_mask_by_points(result, pt_label, kernel_size=5).bool()
        # print(f"iter2 Converged at{iter2}, Diff: {diff}")
        fg_area_new = result
        bg_area_new = ~result
        diff = torch.norm(bg_area_new.float() - bg_area.float())
        if diff < (H * W / 64) ** 0.5:
            converge_time += 1
        else:
            converge_time = 0
            # print(f"iter1 Converged at{iter1}, Diff: {diff}")
        if converge_time > 2:
            break
        #     print(f"iter1 {iter1}, Diff: {diff}")
        if result.float().sum() < 4:
            noise_ratio = noise_ratio * 0.5
            continue
        noise_ratio = noise_ratio * 0.95
        if fg_area_new.sum() > (grad_intensity > 0.1).float().sum() * 2 and (H + W) > 96:
            break
        decay_rate = 0.5
        fg_area, bg_area = fg_area_new.float() + fg_area*decay_rate, bg_area_new.float()+ bg_area*decay_rate
        fg_area, bg_area = torch.clamp(fg_area, min=0.0, max=1.0), torch.clamp(bg_area, min=0.0, max=1.0)

    return result
    
def evolve_target(grad_intensity, target_mask, image, restrain_mask, pt_label):
    """
    Args:
        grad_intensity: 梯度强度 [H, W]
        target_mask: 
        pt_label:
    Returns:
        result_mask: [H, W]
    """
    H, W = grad_intensity.shape
    if target_mask.float().sum() < 4:
        print("initiated one")
        result = initial_target(grad_intensity, pt_label, 0.1)
        return result
    GI = grad_intensity
    image = (image - image.min())/(image.max() - image.min() + 1e-8)
    # 找局部最高值，再梯度强度➗局部最大值，进行局部归一化
    local_max = F.max_pool2d(GI.unsqueeze(0).unsqueeze(0), (5,5), stride=1, padding=2)
    local_norm_image = grad_intensity / (local_max + 1e-11)

    # 局部最大值
    local_max2 = F.max_pool2d(local_norm_image, (3,3), stride=1, padding=1)
    summit_mask = (local_norm_image == local_max2)
    summit_mask = summit_mask.squeeze(0).squeeze(0)
    local_norm_image = local_norm_image.squeeze(0).squeeze(0)

    fg_area = target_mask.float()
    # bg_area = (1-dilate_mask(target_mask.float(), 1)).float()
    bg_area = (1-target_mask.float()).float()
    noise_ratio = 0.05
    converge_time = 0
    for iter1 in range(50):
        fg_mask = fg_area * summit_mask

        bg_mask = torch.zeros_like(GI)
        bg_seed_num = bg_area.sum()/(fg_area.sum()+1)*fg_mask.sum()

        bg_mask = add_uniform_points_v3(GI, bg_area.bool(), bg_mask.bool(), int(bg_seed_num), mode='bg')
        # result_num_ratio = fg_area.sum() / (fg_area.sum() + bg_area.sum())
        for iter2 in range(100):
            noise = torch.rand(GI.shape)
            GI = grad_intensity * (1-noise_ratio) + noise * noise_ratio
            # fig = plt.figure(figsize=(35, 5))
            # plt.subplot(1, 7, 1)
            # plt.imshow(GI.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
            # plt.subplot(1, 7, 2)
            # plt.imshow(fg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
            # plt.subplot(1, 7, 3)
            # plt.imshow(bg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)

            max_num = np.ceil(fg_mask.sum().item())
            min_num = np.ceil(bg_mask.sum().item())

            # mix_ratio = 0.8
            # logits = GI * mix_ratio + image * (1 - mix_ratio)
            # print(GI.shape, image.shape, logits.shape)
            if max_num > 8 and min_num > 8:
                fg_ratio = 1 - max_num /(fg_area.sum() + 1e-8)
                bg_ratio = 1 - min_num /(bg_area.sum() + 1e-8)

                fg_ratio = max(fg_ratio, 0.20)
                bg_ratio = max(bg_ratio, 0.20)

                local_max_num = 9 if max_num * fg_ratio < 9 else max_num * fg_ratio
                _, fg_vwo, _, fg_v = compute_weighted_mean_variance(GI, fg_mask > 0.1, int(local_max_num))

                local_min_num = 9 if min_num * bg_ratio < 9 else min_num * bg_ratio
                _, bg_vwo, _, bg_v = compute_weighted_mean_variance(GI, bg_mask > 0.1, int(local_min_num))
                result_ = fg_v/(fg_vwo+1e-8) -bg_v/(bg_vwo+1e-8)   #(H,W)
                # result_ = bg_v/(fg_v + 1e-8) - bg_vwo/(fg_vwo + 1e-8)
                # print(fg_ratio, int(local_max_num), int(local_min_num))

                # plt.subplot(1, 7, 4)
                # plt.imshow((bg_v/(bg_vwo+1e-8)-1), cmap='gray'), plt.title(f'max:{(bg_v/(bg_vwo+1e-8)-1).max()}')
                # plt.subplot(1, 7, 5)
                # plt.imshow((fg_v/(fg_vwo+1e-8)-1), cmap='gray'), plt.title(f'max:{(fg_v/(fg_vwo+1e-8)-1).max()}')
            else:
                result_ = torch.where(fg_area > 0.1, -GI, GI)
            result_ = keep_negative_by_top3_magnitude_levels(result_, target_size=fg_area.sum())
            result = torch.where(result_ < 0., torch.ones_like(GI), torch.zeros_like(GI)).bool()
            result = filter_mask_by_points(result, target_mask, kernel_size=1).bool()

            # plt.subplot(1, 7, 6)
            # plt.imshow(result, cmap='gray', vmax=1.0, vmin=0.)
            # plt.subplot(1, 7, 7)
            # plt.imshow(result_, cmap='gray')
            # plt.show(block=False)
            # a = input()

            # result__ = result_ * ~((fg_mask>0.1) | (bg_mask>0.1))

            # result_pos = (result__ > 0)*result__
            # result_pos = result__ / (result_.max() + 1e-8)

            # result_neg = (result__ < 0) * (-result__)
            # result_neg = result_neg / (result_neg.max() + 1e-8)
            # order = torch.max(result_pos, result_neg)

            # seedlize_candi = big_num_mask(order, int(H*W*0.1))

            # print(order)
            # print(seedlize_candi)

            # plt.subplot(1, 7, 6)
            # plt.imshow(result_pos, cmap='gray', vmax=1.0, vmin=0.)
            # plt.subplot(1, 7, 7)
            # plt.imshow(result_neg, cmap='gray', vmax=1.0, vmin=0.)
            # plt.show(block=False)
            # a = input()

            fg_seed_num = int(0.1*max_num)
            # fg_seed_num = (seedlize_candi * (fg_area > 0.1)).sum()
            fg_seed_num = fg_seed_num if fg_seed_num > 2 else 2
            fg_mask_new = add_uniform_points_v3(GI, (fg_area > 0.1) * (result_ < 0.), fg_mask>0.1, int(fg_seed_num), mode='fg')
            fg_mask_new = fg_mask_new.bool() * result

            bg_seed_num = int(0.1*min_num)
            # bg_seed_num = (seedlize_candi * (bg_area > 0.1)).sum()
            bg_seed_num = bg_seed_num if bg_seed_num > 2 * bg_area.sum()/fg_area.sum() else 2 * bg_area.sum()/fg_area.sum()
            bg_mask_new = add_uniform_points_v3(GI, (bg_area > 0.1) * (result_ > 0.), bg_mask>0.1, int(bg_seed_num), mode='bg')
            bg_mask_new = bg_mask_new.bool() * ~result

            diff = torch.norm(bg_mask_new.float() - (bg_mask>0.1).float())
            if diff < 1 :
                # print(f"iter1 {iter1} iter2 Converged at {iter2}, Diff: {diff}")
                break
            # else:
            #     print(f"iter1 {iter1} iter2 {iter2}, Diff: {diff}")

            decay_rate = 0.8
            fg_mask, bg_mask = fg_mask_new.float() + fg_mask*decay_rate, bg_mask_new.float() + bg_mask*decay_rate
            # fg_mask, bg_mask = (fg_mask_new.float()/decay_rate + fg_mask)*decay_rate, (bg_mask_new.float()/decay_rate + bg_mask)*decay_rate
            fg_mask, bg_mask = torch.clamp(fg_mask, min=0.0, max=1.0), torch.clamp(bg_mask, min=0.0, max=1.0)
            # fg_mask = torch.where(fg_mask > bg_mask, fg_mask, torch.zeros_like(fg_mask))
            # bg_mask = torch.where(bg_mask >= fg_mask, bg_mask, torch.zeros_like(bg_mask))
        
        result_total = torch.zeros_like(result_)
        fg_ratio, bg_ratio = 0.4, 0.4
        for i in range(10):
            noise = torch.rand(GI.shape)
            GI = grad_intensity * (1-noise_ratio) + noise * noise_ratio

            local_max_num = 9 if max_num * fg_ratio < 9 else max_num * fg_ratio
            _, fg_vwo, _, fg_v = compute_weighted_mean_variance(GI, fg_mask > 0.1, int(local_max_num))

            local_min_num = 9 if min_num * bg_ratio < 9 else min_num * bg_ratio
            _, bg_vwo, _, bg_v = compute_weighted_mean_variance(GI, bg_mask > 0.1, int(local_min_num))
            result_total += fg_v/(fg_vwo+1e-8) -bg_v/(bg_vwo+1e-8)   #(H,W)
            
            fg_ratio -= 0.02
            bg_ratio -= 0.02
        
        result_total = keep_negative_by_top3_magnitude_levels(result_total, target_size=fg_area.sum())
        result = torch.where(result_total < 0., torch.ones_like(GI), torch.zeros_like(GI)).bool()
        result = filter_mask_by_points(result, target_mask, kernel_size=1).bool()

        # fig = plt.figure(figsize=(25, 5))
        # plt.subplot(1, 5, 1)
        # plt.imshow(GI.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
        # plt.subplot(1, 5, 2)
        # plt.imshow(result_total, cmap='gray')
        # plt.subplot(1, 5, 3)
        # plt.imshow(result_total, cmap='gray')
        # plt.show(block=False)
        # plt.subplot(1, 5, 4)
        # plt.imshow(fg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
        # plt.subplot(1, 5, 5)
        # plt.imshow(bg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
        # a = input()

        fg_area_new = result
        bg_area_new = ~result
        # bg_area_new = filter_mask_by_points(~result, lowest_mask, kernel_size=1).bool()
        diff = torch.norm(bg_area_new.float() - bg_area.float())
        if diff < (H * W / 64) ** 0.5:
            # print(f"iter1 Converged at{iter1}, Diff: {diff}")
            converge_time += 1
        # else:
        #     print(f"iter1 {iter1}, Diff: {diff}")
        # fg_area, bg_area = fg_area_new.float(), bg_area_new.float()
        else:
            converge_time = 0
        if converge_time > 2:
            break
        if fg_area_new.sum() > (grad_intensity > 0.1).float().sum() * 2 and (H + W) > 96:
            break
        if result.float().sum() < 4:
            noise_ratio = noise_ratio * 0.5
            continue
        noise_ratio = noise_ratio * 0.95
        decay_rate = 0.5
        fg_area, bg_area = fg_area_new.float() + fg_area*decay_rate, bg_area_new.float()+ bg_area*decay_rate
        # fg_area, bg_area = (fg_area_new.float()/decay_rate + fg_area)*decay_rate, (bg_area_new.float()/decay_rate + bg_area)*decay_rate
        fg_area, bg_area = torch.clamp(fg_area, min=0.0, max=1.0), torch.clamp(bg_area, min=0.0, max=1.0)
        # fg_area = torch.where(fg_area > bg_area, fg_area, torch.zeros_like(fg_area))
        # bg_area = torch.where(bg_area >= fg_area, bg_area, torch.zeros_like(bg_area))

    return result