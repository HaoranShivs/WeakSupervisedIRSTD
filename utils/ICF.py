import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from utils.sum_val_filter import min_pool2d
from utils.utils import compute_mask_pixel_distances_with_coords, extract_local_windows, min_positive_per_local_area, \
    compute_local_extremes, compute_weighted_mean_variance, random_select_from_prob_mask, select_complementary_pixels, \
    get_connected_mask_long_side, keep_negative_by_top2_magnitude_levels, add_uniform_points_grid_cuda, big_num_mask, add_uniform_points_v2, \
    add_uniform_points_v3, add_uniform_points_with_logits, get_min_value_outermost_mask, periodic_function, smooth_optim, bilateral_smooth_logits, \
    compute_weighted_mean_variance_fast, get_distance_matrix_64
from utils.adaptive_filter import filter_mask_by_points, robust_min_max
from utils.refine import dilate_mask, erode_mask

from typing import Tuple, Optional, Dict, List


def initial_target_v1(grad_intensity: torch.Tensor, fg_thre: float = 0.5, bg_thre: float = 0.1):
    """
    Args:
        grad_intensity: 输入图像 [H, W]
        threshold: 阈值，用于初步划分种子点是否为有效种子点，如前景种子点需>threshold，背景种子点需<threshold
    Returns:
        seed_cofidence: 概率图 [H, W, 2]
    """
    grad_intensity = grad_intensity.cuda()
    H, W = grad_intensity.shape
    # Precompute coords_grid once (outside evolve_target or at beginning)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=grad_intensity.device),
        torch.arange(W, device=grad_intensity.device),
        indexing='ij'
    )
    coords_grid = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2).float()  # [HW, 2]

    pixel_num_min = 7
    # 初始化fg_area, bg_area
    fg_area = (grad_intensity > fg_thre).float()
    if fg_area.sum() < pixel_num_min:
        fg_area = (grad_intensity >=grad_intensity.view(-1).sort(descending=True).values[pixel_num_min]).float()
    bg_area = (grad_intensity < bg_thre).float()
    if bg_area.sum() < pixel_num_min:
        bg_area = (grad_intensity <= grad_intensity.view(-1).sort(descending=True).values[-pixel_num_min]).float()

    noise_level = 0.05
    for iter1 in range(20):
        fg_mask = (fg_area > 0.1).float() * (grad_intensity > 0.8)
        bg_mask = (bg_area > 0.1).float()
        for iter2 in range(20):
            noise = torch.rand(grad_intensity.shape, device=grad_intensity.device)
            GI = grad_intensity * (1-noise_level) + noise * noise_level

            max_num = torch.ceil(fg_mask.sum()).int()
            min_num = torch.ceil(bg_mask.sum()).int()

            # === Precompute sparse representation for fg ===
            fg_mask_coords = torch.nonzero(fg_mask, as_tuple=False)  # [N_fg, 2]
            if fg_mask_coords.numel() > 0:
                fg_mask_logits = GI[fg_mask_coords[:, 0], fg_mask_coords[:, 1]]  # [N_fg]
            else:
                fg_mask_logits = torch.empty(0, device=GI.device)

            # === Precompute for bg ===
            bg_mask_coords = torch.nonzero(bg_mask, as_tuple=False)
            if bg_mask_coords.numel() > 0:
                bg_mask_logits = GI[bg_mask_coords[:, 0], bg_mask_coords[:, 1]]
            else:
                bg_mask_logits = torch.empty(0, device=GI.device)

            if max_num >= pixel_num_min and min_num > pixel_num_min:
                fg_ratio = 1 - max_num /(fg_area.sum() + 1e-8)
                bg_ratio = 1 - min_num /(bg_area.sum() + 1e-8)

                fg_ratio = torch.max(fg_ratio, torch.tensor(0.50))
                bg_ratio = torch.max(bg_ratio, torch.tensor(0.50))

                local_max_num = pixel_num_min if max_num * fg_ratio < pixel_num_min else max_num * fg_ratio
                local_min_num = pixel_num_min if min_num * bg_ratio < pixel_num_min else min_num * bg_ratio

                # _, fg_vwo, _, fg_v = compute_weighted_mean_variance(GI, fg_mask > 0.1, int(local_max_num))
                # _, bg_vwo, _, bg_v = compute_weighted_mean_variance(GI, bg_mask > 0.1, int(local_min_num))

                # Now call fast version
                _, fg_vwo, _, fg_v = compute_weighted_mean_variance_fast(
                    GI, fg_mask_coords, fg_mask_logits, coords_grid, 
                    top_k=torch.tensor(local_max_num, dtype=torch.int32), device=GI.device
                )
                _, bg_vwo, _, bg_v = compute_weighted_mean_variance_fast(
                    GI, bg_mask_coords, bg_mask_logits, coords_grid, 
                    top_k=torch.tensor(local_min_num, dtype=torch.int32), device=GI.device
                )

                bg_vg = bg_v/(bg_vwo+1e-8)
                fg_vg = fg_v/(fg_vwo+1e-8)

                result_ = fg_vg - bg_vg   #(H,W)
            else:
                result_ = torch.where(fg_area > 0.1, -GI, GI)
            result_ = keep_negative_by_top2_magnitude_levels(result_, target_size=fg_area.sum())
            result = torch.where(result_ < 0., torch.ones_like(GI), torch.zeros_like(GI)).bool()

            fg_seed_num = int(0.1*max_num)
            fg_seed_num = fg_seed_num if fg_seed_num > 2 else 2
            fg_mask_new = add_uniform_points_v3(grad_intensity, (fg_area > 0.1) * (result_ < 0.), fg_mask>0.1, int(fg_seed_num), mode='fg')
            fg_mask_new = fg_mask_new.bool() * result

            bg_seed_num = int(0.1*min_num)
            bg_seed_num = bg_seed_num if bg_seed_num > 8 else 8
            bg_mask_new = add_uniform_points_grid_cuda((bg_area > 0.1) * (result_ >= 0.), bg_mask>0.1, int(bg_seed_num))
            bg_mask_new = bg_mask_new.bool() * ~result

            diff = torch.norm(fg_mask_new.float() - (fg_mask > 0.1).float()) 
            break_thre = (fg_area > 0.1).float().sum() / 64
            if diff < break_thre:
                # print(f"iter1 {iter1} iter2 Converged at {iter2}")
                break
            # else:
            #     print(f"iter1 {iter1} iter2 {iter2}, Diff: {diff}")

            decay_rate = 0.5
            fg_mask, bg_mask = fg_mask_new.float()*(1-decay_rate) + fg_mask*decay_rate, bg_mask_new.float()*(1-decay_rate) + bg_mask*decay_rate
        
        result_total = torch.zeros_like(result_)
        fg_ratio, bg_ratio = 0.6, 0.6
        for i in range(10):
            noise = torch.rand(GI.shape, device=GI.device)
            GI = grad_intensity * (1-noise_level) + noise * noise_level

            local_max_num = pixel_num_min if max_num * fg_ratio < pixel_num_min else max_num * fg_ratio
            local_min_num = pixel_num_min if min_num * bg_ratio < pixel_num_min else min_num * bg_ratio

            # Now call fast version
            _, fg_vwo, _, fg_v = compute_weighted_mean_variance_fast(
                GI, fg_mask_coords, fg_mask_logits, coords_grid, 
                top_k=int(local_max_num), device=GI.device
            )
            _, bg_vwo, _, bg_v = compute_weighted_mean_variance_fast(
                GI, bg_mask_coords, bg_mask_logits, coords_grid, 
                top_k=int(local_min_num), device=GI.device
            )

            bg_vg = bg_v/(bg_vwo+1e-8)
            fg_vg = fg_v/(fg_vwo+1e-8)

            result_total += fg_vg - bg_vg   #(H,W)
            
            fg_ratio -= 0.02
            bg_ratio -= 0.02

        result_total = keep_negative_by_top2_magnitude_levels(result_total, target_size=fg_area.sum())
        result = torch.where(result_total < 0., torch.ones_like(GI), torch.zeros_like(GI)).bool()

        # print(f"iter1 {iter1} Converged at {iter2}, Diff: {diff}")
        # 修改area
        # result = filter_mask_by_points(result, pt_label, kernel_size=5).bool()
        # print(f"iter2 Converged at{iter2}, Diff: {diff}")
        fg_area_new = result
        bg_area_new = ~result
        diff = torch.norm(fg_area_new.float() - fg_area.float())
        break_thre = (fg_area > 0.1).float().sum() / 64
        if diff < break_thre:
            break
        if result.float().sum() < 4:
            noise_level = noise_level * 0.5
            continue
        noise_level = noise_level * 0.95
        if fg_area_new.sum() > (grad_intensity > 0.1).float().sum() * 2 and (H + W) > 96:
            break
        decay_rate = 0.5
        fg_area, bg_area = fg_area_new.float()*(1-decay_rate) + fg_area*decay_rate, bg_area_new.float()*(1-decay_rate)+ bg_area*decay_rate

    return result.cpu()
    
def initial_target(grad_intensity: torch.Tensor, fg_thre: float = 0.5, bg_thre: float = 0.1):
    """
    Args:
        grad_intensity: 输入图像 [H, W]
        threshold: 阈值，用于初步划分种子点是否为有效种子点，如前景种子点需>threshold，背景种子点需<threshold
    Returns:
        seed_cofidence: 概率图 [H, W, 2]
    """
    # grad_intensity = grad_intensity.cuda()
    H, W = grad_intensity.shape
    # prepare
    full_dist_matrix_4096x4096 = get_distance_matrix_64()
    full_dist_4d = full_dist_matrix_4096x4096.view(64, 64, 64, 64)  # (64,64,64,64)
    dist_sub_4d = full_dist_4d[:H, :W, :H, :W]  # (H, W, H, W)
    dist_sub = dist_sub_4d.reshape(H * W, H * W).to(device=grad_intensity.device)

    pixel_num_min = 7
    # 初始化fg_area, bg_area
    fg_area = (grad_intensity > fg_thre).float()
    if fg_area.sum() < pixel_num_min:
        fg_area = (grad_intensity >=grad_intensity.view(-1).sort(descending=True).values[pixel_num_min]).float()
    bg_area = (grad_intensity < bg_thre).float()
    if bg_area.sum() < pixel_num_min:
        bg_area = (grad_intensity <= grad_intensity.view(-1).sort(descending=True).values[-pixel_num_min]).float()

    noise_level = 0.05
    for iter1 in range(20):
        fg_mask = (fg_area > 0.1).float() * (grad_intensity > 0.8)
        bg_mask = (bg_area > 0.1).float() * (grad_intensity < 0.5)
        for iter2 in range(20):
            noise = torch.rand(grad_intensity.shape, device=grad_intensity.device)
            GI = grad_intensity * (1-noise_level) + noise * noise_level

            max_num = torch.ceil(fg_mask.sum()).int()
            min_num = torch.ceil(bg_mask.sum()).int()

            # fig = plt.figure(figsize=(35, 5))
            # plt.subplot(1, 7, 1)
            # plt.imshow(GI.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
            # plt.subplot(1, 7, 2)
            # plt.imshow(fg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
            # plt.subplot(1, 7, 3)
            # plt.imshow(bg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)

            if max_num >= pixel_num_min and min_num > pixel_num_min:
                fg_ratio = 1 - max_num /(fg_area.sum() + 1e-8)
                bg_ratio = 1 - min_num /(bg_area.sum() + 1e-8)

                fg_ratio = max(fg_ratio, 0.50)
                bg_ratio = max(bg_ratio, 0.50)

                # local_max_num = pixel_num_min if max_num * fg_ratio < pixel_num_min else max_num * fg_ratio
                # local_min_num = pixel_num_min if min_num * bg_ratio < pixel_num_min else min_num * bg_ratio
                # _, fg_vwo, _, fg_v = compute_weighted_mean_variance(GI, fg_mask > 0.1, int(local_max_num))
                # _, bg_vwo, _, bg_v = compute_weighted_mean_variance(GI, bg_mask > 0.1, int(local_min_num))

                # Now call fast version
                # _, fg_vwo, _, fg_v = compute_weighted_mean_variance_fast(GI, (fg_mask>0.1), dist_sub, top_k=int(local_max_num), coeff=1.)
                # _, bg_vwo, _, bg_v = compute_weighted_mean_variance_fast(GI, (bg_mask>0.1), dist_sub, top_k=int(local_min_num), coeff=1.)
                _, fg_vwo, _, fg_v = compute_weighted_mean_variance_fast(GI, (fg_mask>0.1), dist_sub, coeff=3.)
                _, bg_vwo, _, bg_v = compute_weighted_mean_variance_fast(GI, (bg_mask>0.1), dist_sub, coeff=3.)

                bg_vg = bg_v/(bg_vwo+1e-8)
                fg_vg = fg_v/(fg_vwo+1e-8)

                # plt.subplot(1, 7, 4)
                # plt.imshow(bg_vg, cmap='gray')
                # plt.subplot(1, 7, 5)
                # plt.imshow(fg_vg, cmap='gray')

                result_ = fg_vg - bg_vg   #(H,W)
            else:
                result_ = torch.where(fg_area > 0.1, -GI, GI)
            result_ = keep_negative_by_top2_magnitude_levels(result_, target_size=fg_area.sum())
            result = torch.where(result_ < 0., torch.ones_like(GI), torch.zeros_like(GI)).bool()

            # plt.subplot(1, 7, 6)
            # plt.imshow(result_, cmap='gray')
            # plt.subplot(1, 7, 7)
            # plt.imshow(result, cmap='gray')
            # plt.show(block=False)
            # a = input()

            fg_seed_num = int(0.1*max_num)
            fg_seed_num = fg_seed_num if fg_seed_num > 2 else 2
            fg_mask_new = add_uniform_points_v3(grad_intensity, (fg_area > 0.1) * (result_ < 0.), fg_mask>0.1, int(fg_seed_num), mode='fg')
            fg_mask_new = fg_mask_new.bool() * result

            bg_seed_num = int(0.1*min_num)
            bg_seed_num = bg_seed_num if bg_seed_num > 8 else 8
            bg_mask_new = add_uniform_points_grid_cuda((bg_area > 0.1) * (result_ >= 0.), bg_mask>0.1, int(bg_seed_num))
            bg_mask_new = bg_mask_new.bool() * ~result

            diff = torch.norm(fg_mask_new.float() - (fg_mask > 0.1).float()) 
            break_thre = (fg_area > 0.1).float().sum() / 128
            if diff < break_thre:
                # print(f"iter1 {iter1} iter2 Converged at {iter2}, Diff: {diff}, break_thre: {break_thre}")
                break
            # else:
            #     print(f"iter1 {iter1} iter2 {iter2}, Diff: {diff}")

            decay_rate = 0.5
            fg_mask, bg_mask = fg_mask_new.float()*(1-decay_rate) + fg_mask*decay_rate, bg_mask_new.float()*(1-decay_rate) + bg_mask*decay_rate
        
        result_total = torch.zeros_like(result_)
        fg_ratio, bg_ratio = 0.6, 0.6
        for i in range(10):
            noise = torch.rand(GI.shape, device=GI.device)
            GI = grad_intensity * (1-noise_level) + noise * noise_level

            local_max_num = pixel_num_min if max_num * fg_ratio < pixel_num_min else max_num * fg_ratio
            local_min_num = pixel_num_min if min_num * bg_ratio < pixel_num_min else min_num * bg_ratio

            # Now call fast version
            # _, fg_vwo, _, fg_v = compute_weighted_mean_variance_fast(GI, (fg_mask>0.1), dist_sub, top_k=int(local_max_num), coeff=1.)
            # _, bg_vwo, _, bg_v = compute_weighted_mean_variance_fast(GI, (bg_mask>0.1), dist_sub, top_k=int(local_min_num), coeff=1.)
            _, fg_vwo, _, fg_v = compute_weighted_mean_variance_fast(GI, (fg_mask>0.1), dist_sub, coeff=3.)
            _, bg_vwo, _, bg_v = compute_weighted_mean_variance_fast(GI, (bg_mask>0.1), dist_sub, coeff=3.)

            bg_vg = bg_v/(bg_vwo+1e-8)
            fg_vg = fg_v/(fg_vwo+1e-8)

            result_total += fg_vg - bg_vg   #(H,W)
            
            fg_ratio -= 0.02
            bg_ratio -= 0.02

        result_total = keep_negative_by_top2_magnitude_levels(result_total, target_size=fg_area.sum())
        result = torch.where(result_total < 0., torch.ones_like(GI), torch.zeros_like(GI)).bool()

        # print(f"iter1 {iter1} Converged at {iter2}, Diff: {diff}")
        # 修改area
        # result = filter_mask_by_points(result, pt_label, kernel_size=5).bool()
        # print(f"iter2 Converged at{iter2}, Diff: {diff}")
        fg_area_new = result
        bg_area_new = ~result

        # fig = plt.figure(figsize=(25, 5))
        # plt.subplot(1, 5, 1)
        # plt.imshow(GI.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
        # plt.subplot(1, 5, 2)
        # plt.imshow(result, cmap='gray')
        # plt.subplot(1, 5, 3)
        # plt.imshow(result_total, cmap='gray')
        # plt.show(block=False)
        # plt.subplot(1, 5, 4)
        # plt.imshow(fg_area.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
        # plt.subplot(1, 5, 5)
        # plt.imshow(fg_area_new.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
        # a = input()

        diff = torch.norm(fg_area_new.float() - fg_area.float())
        break_thre = (fg_area > 0.1).float().sum() / 64
        if diff < break_thre:
            break
        if result.float().sum() < 4:
            noise_level = noise_level * 0.5
            continue
        noise_level = noise_level * 0.95
        if fg_area_new.sum() > (grad_intensity > 0.1).float().sum() * 2 and (H + W) > 96:
            break
        decay_rate = 0.5
        fg_area, bg_area = fg_area_new.float()*(1-decay_rate) + fg_area*decay_rate, bg_area_new.float()*(1-decay_rate)+ bg_area*decay_rate

    return result.cpu()
    

def evolve_target(grad_intensity, target_mask, image, pt_label, alpha, beta):
    """
    Args:
        grad_intensity: 梯度强度 [H, W]
        target_mask: 
        pt_label:
    Returns:
        result_mask: [H, W]
    """
    # grad_intensity = grad_intensity.cuda()
    H, W = grad_intensity.shape
    target_mask = target_mask > 0.5
    # prepare
    full_dist_matrix_4096x4096 = get_distance_matrix_64()
    full_dist_4d = full_dist_matrix_4096x4096.view(64, 64, 64, 64)  # (64,64,64,64)
    dist_sub_4d = full_dist_4d[:H, :W, :H, :W]  # (H, W, H, W)
    dist_sub = dist_sub_4d.reshape(H * W, H * W).to(device=grad_intensity.device)

    pixel_num_min = 7

    # 噪声水平测量
    # noise_level_ = grad_intensity[target_mask.bool()].min()
    noise_level_ = 0.5 * (alpha * 0.1 + (1-alpha)*(1-beta)) / (1-(1-alpha)*beta + 1e-8)
    noise_level = torch.clamp(torch.tensor(noise_level_), min=0.05, max=0.051)

    fg_area = target_mask.float()
    bg_area = 1-target_mask.float()
    pixel_num_min = 7
    for iter1 in range(20):
        fg_mask = (fg_area > 0.1).float() * (grad_intensity > 0.8)
        bg_mask = (bg_area > 0.1).float() * (grad_intensity < 0.5)
        for iter2 in range(20):
            noise = torch.rand(grad_intensity.shape, device=grad_intensity.device)
            GI = grad_intensity * (1-noise_level) + noise * noise_level

            max_num = torch.ceil(fg_mask.sum()).int()
            min_num = torch.ceil(bg_mask.sum()).int()

            # fig = plt.figure(figsize=(35, 5))
            # plt.subplot(1, 7, 1)
            # plt.imshow(GI.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
            # plt.subplot(1, 7, 2)
            # plt.imshow(fg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
            # plt.subplot(1, 7, 3)
            # plt.imshow(bg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)

            if max_num >= pixel_num_min and min_num > pixel_num_min:
                fg_ratio = 1 - max_num /(fg_area.sum() + 1e-8)
                bg_ratio = 1 - min_num /(bg_area.sum() + 1e-8)

                fg_ratio = max(fg_ratio, 0.50)
                bg_ratio = max(bg_ratio, 0.30)

                local_max_num = pixel_num_min if max_num * fg_ratio < pixel_num_min else max_num * fg_ratio
                local_min_num = pixel_num_min if min_num * bg_ratio < pixel_num_min else min_num * bg_ratio

                # _, fg_vwo, _, fg_v = compute_weighted_mean_variance(GI, fg_mask > 0.1, int(local_max_num))
                # _, bg_vwo, _, bg_v = compute_weighted_mean_variance(GI, bg_mask > 0.1, int(local_min_num))

                # Now call fast version
                # _, fg_vwo, _, fg_v = compute_weighted_mean_variance_fast(GI, (fg_mask>0.1), dist_sub, top_k=int(local_max_num), coeff=1.)
                # _, bg_vwo, _, bg_v = compute_weighted_mean_variance_fast(GI, (bg_mask>0.1), dist_sub, top_k=int(local_min_num), coeff=1.)
                _, fg_vwo, _, fg_v = compute_weighted_mean_variance_fast(GI, (fg_mask>0.1), dist_sub, coeff=3.)
                _, bg_vwo, _, bg_v = compute_weighted_mean_variance_fast(GI, (bg_mask>0.1), dist_sub, coeff=3.)
                
                bg_vg = bg_v/(bg_vwo+1e-8) - 1
                fg_vg = fg_v/(fg_vwo+1e-8) - 1

                # plt.subplot(1, 7, 4)
                # plt.imshow(bg_vg, cmap='gray')
                # plt.subplot(1, 7, 5)
                # plt.imshow(fg_vg, cmap='gray')

                result_ = fg_vg - bg_vg   #(H,W)
            else:
                result_ = torch.where(fg_area > 0.1, -GI, GI)
            result_ = keep_negative_by_top2_magnitude_levels(result_, target_size=fg_area.sum())
            result = torch.where(result_ < 0., torch.ones_like(GI), torch.zeros_like(GI)).bool()
            result = filter_mask_by_points(result, target_mask, kernel_size=1).bool()

            fg_seed_num = int(0.1*max_num)
            fg_seed_num = fg_seed_num if fg_seed_num > 2 else 2
            fg_mask_new = add_uniform_points_v3(grad_intensity, (fg_area > 0.1) * (result_ < 0.), fg_mask>0.1, int(fg_seed_num), mode='fg')
            fg_mask_new = fg_mask_new.bool() * result

            bg_seed_num = int(0.1*min_num)
            bg_seed_num = bg_seed_num if bg_seed_num > 2 else 2
            bg_mask_new = add_uniform_points_grid_cuda((bg_area > 0.1) * (result_ > 0.), bg_mask>0.1, int(bg_seed_num))
            bg_mask_new = bg_mask_new.bool() * ~result

            # plt.subplot(1, 7, 6)
            # plt.imshow(result_, cmap='gray', vmax=1.0, vmin=0.)
            # plt.subplot(1, 7, 7)
            # plt.imshow(result, cmap='gray')
            # plt.show(block=False)
            # a = input()

            diff = torch.norm(fg_mask_new.float() - (fg_mask > 0.1).float()) 
            break_thre = (fg_area > 0.1).float().sum() / 128
            if diff < break_thre:
                # print(f"iter1 {iter1} iter2 Converged at {iter2}")
                break
            # else:
            #     print(f"iter1 {iter1} iter2 {iter2}, Diff: {diff}")

            decay_rate = 0.5
            fg_mask, bg_mask = fg_mask_new.float()*(1-decay_rate) + fg_mask*decay_rate, bg_mask_new.float()*(1-decay_rate) + bg_mask*decay_rate
        
        result_total = torch.zeros_like(result_)
        fg_ratio, bg_ratio = 0.6, 0.4
        for i in range(10):
            noise = torch.rand(GI.shape)
            GI = grad_intensity * (1-noise_level) + noise * noise_level

            local_max_num = pixel_num_min if max_num * fg_ratio < pixel_num_min else max_num * fg_ratio
            local_min_num = pixel_num_min if min_num * bg_ratio < pixel_num_min else min_num * bg_ratio
            
            # _, fg_vwo, _, fg_v = compute_weighted_mean_variance(GI, fg_mask > 0.1, int(local_max_num))
            # _, bg_vwo, _, bg_v = compute_weighted_mean_variance(GI, bg_mask > 0.1, int(local_min_num))

            # Now call fast version
            # _, fg_vwo, _, fg_v = compute_weighted_mean_variance_fast(GI, (fg_mask>0.1), dist_sub, top_k=int(local_max_num), coeff=1.)
            # _, bg_vwo, _, bg_v = compute_weighted_mean_variance_fast(GI, (bg_mask>0.1), dist_sub, top_k=int(local_min_num), coeff=1.)
            _, fg_vwo, _, fg_v = compute_weighted_mean_variance_fast(GI, (fg_mask>0.1), dist_sub, coeff=3.)
            _, bg_vwo, _, bg_v = compute_weighted_mean_variance_fast(GI, (bg_mask>0.1), dist_sub, coeff=3.)

            bg_vg = bg_v/(bg_vwo+1e-8)
            fg_vg = fg_v/(fg_vwo+1e-8)

            result_total += fg_vg - bg_vg   #(H,W)
            
            fg_ratio -= 0.02
            bg_ratio -= 0.02
        
        result_total_ = keep_negative_by_top2_magnitude_levels(result_total, target_size=fg_area.sum())
        result = torch.where(result_total_ < 0., torch.ones_like(GI), torch.zeros_like(GI)).bool()
        result = filter_mask_by_points(result, target_mask, kernel_size=1).bool()

        # fig = plt.figure(figsize=(25, 5))
        # plt.subplot(1, 5, 1)
        # plt.imshow(GI.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
        # plt.subplot(1, 5, 2)
        # plt.imshow(result, cmap='gray')
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
        diff = torch.norm(fg_area_new.float() - fg_area.float())
        break_thre = (fg_area > 0.1).float().sum() / 128
        if diff < break_thre:
            break
        if fg_area_new.sum() > (grad_intensity > 0.1).float().sum() * 2 and (H + W) > 96:
            break
        if result.float().sum() < 4:
            noise_level = noise_level * 0.5
            continue
        noise_level = noise_level * 0.95
        decay_rate = 0.5
        fg_area, bg_area = fg_area_new.float()*(1-decay_rate) + fg_area*decay_rate, bg_area_new.float()*(1-decay_rate)+ bg_area*decay_rate
        fg_area, bg_area = torch.clamp(fg_area, min=0.0, max=1.0), torch.clamp(bg_area, min=0.0, max=1.0)
    return result