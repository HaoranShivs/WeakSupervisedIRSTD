import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

from utils.sum_val_filter import min_pool2d
from utils.utils import compute_mask_pixel_distances_with_coords, extract_local_windows, min_positive_per_local_area, \
    compute_local_extremes, compute_weighted_mean_variance, random_select_from_prob_mask, select_complementary_pixels, \
    get_connected_mask_long_side, keep_negative_by_top3_magnitude_levels, add_uniform_points_cuda, big_num_mask, add_uniform_points_v2, \
    add_uniform_points_v3, get_min_value_outermost_mask, periodic_function
from utils.adaptive_filter import filter_mask_by_points
from utils.refine import dilate_mask, erode_mask

from typing import Tuple, Optional, Dict, List


def random_walk_segmentation(image, seeds, beta=90):

    # 构建图的邻接矩阵
    height, width = image.shape
    num_pixels = height * width
    adjacency_matrix = scipy.sparse.lil_matrix((num_pixels, num_pixels))

    for i in range(height):
        for j in range(width):
            pixel_index = i * width + j
            if i > 0:
                neighbor_index = (i - 1) * width + j
                weight = np.exp(-beta * (image[i, j] - image[i - 1, j]) ** 2)
                adjacency_matrix[pixel_index, neighbor_index] = weight
                adjacency_matrix[neighbor_index, pixel_index] = weight
            if j > 0:
                neighbor_index = i * width + (j - 1)
                weight = np.exp(-beta * (image[i, j] - image[i, j - 1]) ** 2)
                adjacency_matrix[pixel_index, neighbor_index] = weight
                adjacency_matrix[neighbor_index, pixel_index] = weight

    # 构建拉普拉斯矩阵
    degree_matrix = scipy.sparse.diags(adjacency_matrix.sum(axis=1).A1)
    laplacian_matrix = degree_matrix - adjacency_matrix

    # 求解线性方程组
    b = np.zeros(num_pixels)
    for seed in seeds:
        b[seed] = 1
    x = scipy.sparse.linalg.spsolve(laplacian_matrix, b)

    # 根据概率分配标签
    segmentation = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            pixel_index = i * width + j
            segmentation[i, j] = 1 if x[pixel_index] > 0.5 else 0

    return segmentation


class RandomWalkPixelLabeling(nn.Module):
    def __init__(self, 
                 spatial_weight: float = 1.0,
                 intensity_weight: float = 1.0,
                 feature_weights: Optional[Dict[str, float]] = None,
                 max_iterations1: int = 10,
                 max_iterations2: int = 100,
                 convergence_threshold: float = 1e-4,
                 device: str = 'cpu'):
        """
        增强版随机游走像素级伪标签生成器
        
        Args:
            spatial_weight: 空间距离权重
            intensity_weight: 像素强度权重
            feature_weights: 其他特征的权重字典
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            device: 计算设备
        """
        super(RandomWalkPixelLabeling, self).__init__()
        self.spatial_weight = spatial_weight
        self.intensity_weight = intensity_weight
        self.feature_weights = feature_weights or {}
        self.max_iter1 = max_iterations1
        self.max_iter2 = max_iterations2
        self.convergence_threshold = convergence_threshold
        self.device = device
        
    def compute_transition_probabilities(self, 
                                        image: torch.Tensor,
                                        flat_boundary_thre: torch.Tensor,
                                        flat_boundary_addition: torch.Tensor,
                                        additional_features: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        计算转移概率矩阵
        
        Args:
            image: 输入图像 [H, W] 或 [H, W, C]
            additional_features: 额外特征字典 {'feature_name': tensor[H, W, feature_dim]}
            
        Returns:
            转移概率矩阵 [H*W, H*W]
        """
        H, W = image.shape[:2]
        N = H * W
        
        # 展平图像
        if len(image.shape) == 3:
            flat_image = image.reshape(N, -1)  # [N, C]
        else:
            flat_image = image.reshape(N, 1)   # [N, 1]
            
        # 构建邻接关系
        adjacency = self._build_adjacency_matrix(H, W)
        
        # 计算基础概率权重
        probabilities = torch.zeros(N, N, device=self.device)
        
        # 获取邻接点对
        row_indices, col_indices = torch.where(adjacency > 0.)
        
        # 计算各种权重的贡献
        weights = self._compute_edge_weights(
            row_indices, col_indices, 
            flat_image, 
            H, W,
            flat_boundary_thre,
            flat_boundary_addition,
            additional_features
        )
        
        # 应用权重到邻接矩阵
        probabilities[row_indices, col_indices] = weights
        
        # 归一化行概率
        row_sum = torch.sum(probabilities, dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0  # 避免除零
        probabilities = probabilities / row_sum

        # target_row_idx, target_col_idx = 4, 4
        # target_idx = target_row_idx * W + target_col_idx
        # print("transition_matrix of target_summit\n", probabilities[target_idx].view(H, W))
        
        return probabilities
    
    def _build_adjacency_matrix(self, H: int, W: int) -> torch.Tensor:
        """构建邻接矩阵（4邻域）"""
        N = H * W
        adjacency = torch.zeros(N, N, device=self.device)
        
        # 4邻域：上、下、左、右
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # # 8邻域：上、下、左、右、左上、左下、右上、右下
        # directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        neighbor_idx = ni * W + nj
                        adjacency[idx, neighbor_idx] = 1.0
        return adjacency
    
    def _compute_edge_weights(self,
                            row_indices: torch.Tensor,
                            col_indices: torch.Tensor,
                            flat_image: torch.Tensor,
                            H: int, W: int,
                            flat_boundary_thre: torch.Tensor,
                            flat_boundary_addition: torch.Tensor,
                            additional_features: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        计算边权重（联合概率）
        """
        # 空间距离权重
        spatial_weights = self._compute_spatial_weights(row_indices, col_indices, H, W)

        # 像素强度权重
        intensity_weights = self._compute_intensity_weights(row_indices, col_indices, flat_image, flat_boundary_thre, flat_boundary_addition, H, W)

        # 综合权重计算
        total_weights = torch.zeros_like(spatial_weights)
        
        cofficient = 5.0
        # 应用空间权重
        if self.spatial_weight > 0:
            total_weights += torch.exp(-spatial_weights * cofficient) * self.spatial_weight
            
        # 应用强度权重
        if self.intensity_weight > 0:
            total_weights += torch.exp(-intensity_weights * cofficient) * self.intensity_weight
            # # 计算基础概率权重
            # probabilities = torch.zeros(H*W, H*W, device=self.device)
            # # 应用权重到邻接矩阵
            # probabilities[row_indices, col_indices] = total_weights

            # target_row_idx, target_col_idx = 13, 14
            # target_idx = target_row_idx * W + target_col_idx
            # print(probabilities[target_idx, target_idx - W - 1], probabilities[target_idx, target_idx - W], probabilities[target_idx, target_idx - W + 1])
            # print(probabilities[target_idx, target_idx - 1], probabilities[target_idx, target_idx], probabilities[target_idx, target_idx + 1])
            # print(probabilities[target_idx, target_idx + W - 1], probabilities[target_idx, target_idx + W], probabilities[target_idx, target_idx + W + 1])

            
        # 应用额外特征权重
        if additional_features and self.feature_weights:
            for feature_name, weight in self.feature_weights.items():
                if feature_name in additional_features:
                    feature_weights = self._compute_feature_weights(
                        row_indices, col_indices, 
                        additional_features[feature_name].reshape(-1, additional_features[feature_name].shape[-1])
                    )
                    total_weights += torch.exp(-feature_weights * cofficient) * weight
        
        return total_weights/(total_weights.max() + 1e-11)
    
    def _compute_spatial_weights(self, row_indices: torch.Tensor, 
                               col_indices: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """计算空间距离权重"""
        # 将线性索引转换为2D坐标
        row_i, row_j = row_indices // W, row_indices % W
        col_i, col_j = col_indices // W, col_indices % W
        
        # 计算欧几里得距离
        distances = torch.sqrt((row_i - col_i).float()**2 + (row_j - col_j).float()**2)
        return distances
    
    def _compute_intensity_weights(self, row_indices: torch.Tensor, col_indices: torch.Tensor, 
                                    flat_image: torch.Tensor, flat_boundary_thre: torch.Tensor,
                                    flat_boundary_addition: torch.Tensor, H, W) -> torch.Tensor:
        """计算像素强度差异权重"""
        # print(flat_image.shape, row_indices.shape, flat_boundary_thre.shape)
        # print(flat_boundary_thre[row_indices].shape)
        # print(flat_image[row_indices].shape)
        # print(flat_image[col_indices].shape)
        high_val_mask = flat_image >= flat_boundary_thre
        low_val_mask = flat_image < flat_boundary_thre
        image_high_val = high_val_mask * flat_image
        image_low_val = low_val_mask * flat_image

        max_limits = compute_local_extremes(flat_image.reshape(H, W), low_val_mask.reshape(H, W), mode='max', local_size=5) #(H,W)
        min_limits = compute_local_extremes(flat_image.reshape(H, W), high_val_mask.reshape(H, W), mode='min', local_size=3)

        image_high_val = (image_high_val - min_limits.reshape(H*W, 1)) / (1 - min_limits.reshape(H*W, 1) + 1e-11)
        image_low_val = image_low_val / (max_limits.reshape(H*W, 1) + 1e-11)

        # flat_boundary_addition = min_limits / (max_limits + 1e-8)

        flat_image_ = image_high_val + image_low_val + flat_boundary_addition * high_val_mask

        diff = torch.abs(flat_image_[row_indices] - flat_image_[col_indices])

        diff = diff.view(-1)
        return diff / (diff.max() + 1e-11)
    
    def _compute_feature_weights(self, row_indices: torch.Tensor, 
                               col_indices: torch.Tensor, 
                               flat_features: torch.Tensor) -> torch.Tensor:
        """计算额外特征差异权重"""
        diff = flat_features[row_indices] - flat_features[col_indices]
        diff = torch.norm(diff, dim=1)
        return diff / (diff.max() + 1e-11)
    
    def random_walk_inference_v1(self, 
                            image: torch.Tensor,
                            seeds: torch.Tensor,
                            additional_features: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        执行随机游走推理生成伪标签
        
        Args:
            image: 输入图像 [H, W] 或 [H, W, C]
            seeds: 种子点标签 [H, W] (0表示未标记，>0表示类别标签)
            additional_features: 额外特征字典
            
        Returns:
            伪标签概率 [H, W, num_classes]
        """
        H, W = image.shape[:2]
        N = H * W
        
        # 获取唯一标签（排除0）
        unique_labels = torch.unique(seeds)
        unique_labels = unique_labels[unique_labels > 0]
        num_classes = len(unique_labels)
        
        if num_classes == 0:
            raise ValueError("No seed labels provided!")
        
        # 创建标签映射（将原始标签映射到连续索引）
        label_mapping = {int(label): idx for idx, label in enumerate(unique_labels)}
        
        # 构建概率矩阵
        P = self.compute_transition_probabilities(image, seeds, additional_features)
        
        # 初始化概率矩阵
        probabilities = torch.zeros(N, num_classes, device=self.device)
        seed_mask = seeds.reshape(-1) > 0
        
        # 设置种子点概率
        for original_label, mapped_idx in label_mapping.items():
            mask = (seeds.reshape(-1) == original_label)
            probabilities[mask, mapped_idx] = 1.0
        
        # 迭代更新概率 
        prev_probabilities = probabilities.clone()
        for iteration in range(self.max_iterations):
            # 更新非种子点的概率
            non_seed_mask = ~seed_mask
            
            if non_seed_mask.sum() > 0:
                # 只更新非种子点
                P_non_seed = P[non_seed_mask][:, non_seed_mask]
                P_seed_to_non = P[non_seed_mask][:, seed_mask]
                
                # 计算从种子点传递的概率
                seed_probabilities = probabilities[seed_mask]
                seed_contribution = torch.matmul(P_seed_to_non, seed_probabilities)
                
                # 计算从非种子点传递的概率
                non_seed_contribution = torch.matmul(P_non_seed, probabilities[non_seed_mask])
                
                # 更新非种子点概率
                new_probabilities = 0.5 * (seed_contribution + non_seed_contribution)
                probabilities[non_seed_mask] = new_probabilities
            
            # 检查收敛性
            diff = torch.norm(probabilities - prev_probabilities)
            if diff < self.convergence_threshold:
                print(f"Converged at iteration {iteration}")
                break
                
            prev_probabilities = probabilities.clone()
        
        # 返回最终概率图
        return probabilities.reshape(H, W, num_classes)
    
    def _proper_boundary_threshold(self, image: torch.Tensor, target_mask: torch.Tensor, local_area_size: int=3) -> torch.Tensor:
        """
        根据image和target_mask，生成在local域内，由最小差距的target值和bg值之间的均值作为边界阈值。
        """
        image_l = extract_local_windows(image, local_area_size)
        target_mask_l = extract_local_windows(target_mask, local_area_size)

        image_l_fg = image_l * target_mask_l    # [H, W, local_area_size, local_area_size]
        image_l_bg = image_l * (1 - target_mask_l)  # [H, W, local_area_size, local_area_size]

        image_l_fg_min = min_positive_per_local_area(image_l_fg)    #(H,W)
        image_l_bg_max = image_l_bg.amax(dim=[2,3])  #(H,W)

        valid_mask = (target_mask_l.sum(dim=[2,3]) > 0) * (target_mask_l.sum(dim=[2,3]) < local_area_size*local_area_size)

        boundary_thre = (image_l_fg_min + image_l_bg_max) / 2
        return boundary_thre * valid_mask

    def _proper_boundary_addition(self, probs: torch.Tensor, target_mask: torch.Tensor, local_area_size: int=3) -> torch.Tensor:
        """
        根据image和target_mask，生成在local域内，
        """
        image_l = extract_local_windows(probs, local_area_size)
        target_mask_l = extract_local_windows(target_mask, local_area_size)

        image_l_fg = image_l * target_mask_l    # [H, W, local_area_size, local_area_size]
        image_l_bg = image_l * (1 - target_mask_l)  # [H, W, local_area_size, local_area_size]

        image_l_fg_max = image_l_fg.amax(dim=[2,3])
        image_l_bg_max = image_l_bg.amax(dim=[2,3])
        image_l_bg_min = image_l_bg.amin(dim=[2,3])
        image_l_fg_min = min_positive_per_local_area(image_l_fg)

        mid_idx = local_area_size // 2
        boundary_addition_fg = image_l_fg_max + image_l_bg_max - 2 * image_l[:,:,mid_idx, mid_idx]
        boundary_addition_bg = 2 * image_l[:,:,mid_idx, mid_idx] - image_l_bg_min -image_l_fg_min

        boundary_addition = torch.where(target_mask_l[:,:,mid_idx, mid_idx].bool(), boundary_addition_fg, boundary_addition_bg)

        valid_mask = (target_mask_l.sum(dim=[2,3]) > 0) * (target_mask_l.sum(dim=[2,3]) < local_area_size*local_area_size)

        # 正数就好，负数没必要
        positive_mask = boundary_addition > 0

        return boundary_addition * valid_mask * positive_mask

    def random_walk_inference(self, image: torch.Tensor, seeds: torch.Tensor, additional_features) -> torch.Tensor:
        """
        随机游走推理实现
        """
        H, W = image.shape[:2]
        N = H * W
        
        # 获取唯一标签
        num_classes = seeds.shape[-1]  
        if num_classes == 0:
            raise ValueError("No seed labels provided!")
        # 构建连接矩阵
        P = self.compute_transition_probabilities(image, torch.ones((N,1))*0.1, torch.ones((N,1))*2.0,
                                                  additional_features)  #(N,N)
        
        # 初始化概率矩阵 [N, num_classes]
        probabilities = seeds.reshape(N, num_classes)
        # print(probabilities)

        prev_seeds = probabilities.clone()
        curr_seeds = probabilities.clone()
        
        seeds_masks = curr_seeds > 1e-4   # [N, num_classes]

        # 分离转移概率矩阵
        # P_nn: 非种子点到非种子点的转移概率
        # P_ns: 非种子点到种子点的转移概率
        seed_mask = seeds_masks.sum(dim=1) > 0.  #(N,)
        non_seed_indices = torch.where(~seed_mask)[0]
        seed_indices = torch.where(seed_mask)[0]
        
        if len(non_seed_indices) == 0:
            # 所有点都是种子点
            return probabilities.reshape(H, W, num_classes)
        
        P_nn = P[non_seed_indices][:, non_seed_indices]
        P_ns = P[non_seed_indices][:, seed_indices]
          
        # 迭代更新非种子点概率
        current_probs = probabilities.clone()
        prev_probs = current_probs.clone()

        # print(image)
        curr_seeds_4_show = torch.where(curr_seeds[:,0] > curr_seeds[:,1], curr_seeds[:,0], -curr_seeds[:,1])
        # print(curr_seeds_4_show.view(H, W))

        for iter1 in range(self.max_iter1):
        
            for iter2 in range(self.max_iter2):
                # 只更新非种子点的概率
                # 新概率 = 从非种子点邻居传递的概率 + 从种子点邻居传递的概率
                seed_contrib = torch.matmul(P_ns, current_probs[seed_indices])  # 固定的种子贡献
                non_seed_contrib = torch.matmul(P_nn, current_probs[non_seed_indices])  # 来自非种子点的贡献
                
                # 更新非种子点概率 
                current_probs[non_seed_indices] = seed_contrib + non_seed_contrib
                # current_probs = torch.matmul(P, current_probs)  # 起点是P的行号，终点是current_probs的列号

                # print(f'iter1: {iter1}, iter2: {iter2}')
                # target_row_idx, target_col_idx = 13, 14
                # current_probs_4_show_2 = current_probs.view(H, W, 2)
                # print(current_probs_4_show_2[target_row_idx - 2: target_row_idx + 3, target_col_idx - 2: target_col_idx + 3, 0])
                # print(current_probs_4_show_2[target_row_idx - 2: target_row_idx + 3, target_col_idx - 2: target_col_idx + 3, 1])
                # a = input()

                # title = f'iter1: {iter1}, iter2: {iter2}'
                # current_probs_4_show = torch.where(current_probs[:,0] > current_probs[:,1], current_probs[:,0], -current_probs[:,1])
                # fig = plt.figure()
                # plt.imshow(current_probs_4_show.view(H, W), cmap='gray', vmax=1.0, vmin=-1.0)
                # plt.title(title)
                # plt.show()

                
                # 检查收敛性
                diff = torch.norm(current_probs - prev_probs)
                if diff < self.convergence_threshold:
                    # print(f"iter1_{iter1} Converged at iter2 {iter2}")
                    break
                prev_probs = current_probs.clone()

            print(f'iter1: {iter1}, iter2: {iter2}')
            current_probs_4_show = torch.where(current_probs[:,0] > current_probs[:,1], current_probs[:,0], -current_probs[:,1])
            # print(current_probs_4_show.view(H, W))

            # target_row_idx, target_col_idx = 13, 17
            # current_probs_4_show_2 = current_probs.view(H, W, 2)
            # print(current_probs_4_show_2[target_row_idx - 1: target_row_idx + 2, target_col_idx - 1: target_col_idx + 2, 0])
            # print(current_probs_4_show_2[target_row_idx - 1: target_row_idx + 2, target_col_idx - 1: target_col_idx + 2, 1])
    
            # 利用收敛的非种子点概率更新seeds的置信度。备选项，根据距离再进行优化。
            current_probs_ = current_probs / (current_probs.max(dim=1, keepdim=True).values + 1e-8)
            current_probs_ = torch.matmul(P, current_probs_)

            # debug
            target_row_idx, target_col_idx = 4, 4
            current_probs_4_show_2 = current_probs_.view(H, W, 2)
            print(current_probs_4_show_2[target_row_idx - 1: target_row_idx + 2, target_col_idx - 1: target_col_idx + 2, 0])
            print(current_probs_4_show_2[target_row_idx - 1: target_row_idx + 2, target_col_idx - 1: target_col_idx + 2, 1])

            # 更新种子的选取
            seed_mask_larger = F.max_pool2d(seeds_masks.permute(1, 0).reshape(-1, H, W,).float(), kernel_size=3, stride=1, padding=1).reshape(-1, N).permute(1, 0)     # (N, num_classes)
            seed_grow_thre = 0.5
            curr_seeds_mask = (current_probs_ < seed_grow_thre)  #(N, num_classes)
            curr_seeds_mask[:,[0,1]] = curr_seeds_mask[:,[1,0]]
            curr_seeds_mask = curr_seeds_mask * seed_mask_larger
            new_only_seeds_mask = torch.clamp_min(curr_seeds_mask.float() - seeds_masks.float(), 0.)

            update_rate = 0.5
            curr_seeds = (1 - update_rate) * curr_seeds + update_rate * current_probs_ * seeds_masks \
                + new_only_seeds_mask * current_probs_ #(N, num_classes)
            curr_seeds = curr_seeds / (curr_seeds.max() + 1e-8)
            seeds_masks = curr_seeds > 1e-4

            # print(image)
            curr_seeds_4_show = torch.where(curr_seeds[:,0] > curr_seeds[:,1], curr_seeds[:,0], -curr_seeds[:,1])
            # print(curr_seeds_4_show.view(H, W))

            # 检查收敛性
            diff = torch.norm(curr_seeds - prev_seeds)
            if diff < (self.convergence_threshold * N / seed_mask.float().sum()):
                print(f"Converged at iter1 {iter1}, iter2 {iter2}")
                break
            prev_seeds = curr_seeds.clone()
            current_probs[seeds_masks] = curr_seeds[seeds_masks]
            
            target_mask = current_probs[:, 0] > current_probs[:, 1] #(N,)
            target_mask = target_mask.view(H, W)

            # 更新背景->目标的建议隔离阈值。
            current_boundary_thre = self._proper_boundary_threshold(image, target_mask.float(), local_area_size=3)
            # print('current_boundary_thre:\n', current_boundary_thre)

            # 更新为了提高由上面current_boundary_thre的分割的背景像素和目标像素的距离而加上的偏移
            current_boundary_addition = self._proper_boundary_addition(image, target_mask.float(), local_area_size=3)
            # print('current_boundary_addition:\n', current_boundary_addition)

            # 更新连接矩阵
            P = self.compute_transition_probabilities(image, current_boundary_thre.view(N, -1), torch.ones((N,1))*2.0, additional_features)

            seed_mask = curr_seeds.sum(dim=1) > 0.1  #(N,)
            non_seed_indices = torch.where(~seed_mask)[0]
            seed_indices = torch.where(seed_mask)[0]
            
            P_nn = P[non_seed_indices][:, non_seed_indices]
            P_ns = P[non_seed_indices][:, seed_indices]

            fig = plt.figure(figsize=(20, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(current_probs_4_show.view(H, W), cmap='gray', vmax=1.0, vmin=-1.0)
            plt.subplot(1, 4, 2)
            plt.imshow(curr_seeds_4_show.view(H, W), cmap='gray', vmax=1.0, vmin=-1.0)
            plt.subplot(1, 4, 3)
            # current_boundary_addition[0,0] = 1.0
            plt.imshow(current_boundary_addition, cmap='gray')
            # plt.subplot(1, 4, 4)
            # target_row_idx, target_col_idx = 13, 14
            # target_idx = target_row_idx * W + target_col_idx
            # plt.imshow(P[target_idx].view(H, W), cmap='gray', vmax=1.0, vmin=0.)
            plt.show(block=False)

            a = input()

        return current_probs.reshape(H, W, num_classes)

    def _local_max_min_seed(self, image: torch.Tensor, threshold: float = 0.1):
        """
        获取局部最大值作为前景种子点，局部最小值作为背景种子点，并赋予其作为各自类别的概率
        Args:
            image: 输入图像 [H, W]
            threshold: 阈值，用于初步划分种子点是否为有效种子点，如前景种子点需>threshold，背景种子点需<threshold
        Returns:
            seed_cofidence: 概率图 [H, W, 2]
        """
        image = image.unsqueeze(0).unsqueeze(0)
        ## 前景种子
        # 找局部最高值，再梯度强度➗局部最大值，进行局部归一化
        local_max = F.max_pool2d(image, (5,5), stride=1, padding=2)
        local_norm_image = image / (local_max + 1e-11)

        # 继而再找局部最大值
        local_max2 = F.max_pool2d(local_norm_image, (3,3 ), stride=1, padding=1)
        summit_mask = (local_norm_image == local_max2) * (image > threshold)
        
        summit_p = image
        summit_p = (summit_p * 0.5 + 0.5) * summit_mask # 映射到[0.5, 1]

        # 根据空间位置的聚集程度优化概率
        if summit_mask.sum() > 1:
            coors, dists = compute_mask_pixel_distances_with_coords(summit_mask.squeeze(0).squeeze(0))  # [n, 2]， [n, n]
            # print("summit_dist\n", dists)
            summit_ps = summit_p[0, 0, coors[:,0], coors[:,1]]
            summit_ps = torch.mean(summit_ps * (1-dists), dim=1)    # [n,]
            summit_p[0, 0, coors[:, 0], coors[:, 1]] = summit_ps
        summit_p = summit_p / summit_p.max()

        ## 背景种子
        # 找局部最小值
        local_min = min_pool2d(image, (5,5), stride=1, padding=2)
        valley_mask = (image == local_min) * (image <= threshold)

        valley_p = image 
        valley_p = ((threshold - valley_p) * 0.5 / threshold + 0.5) * valley_mask

        # 根据空间位置的聚集程度优化概率
        if valley_mask.sum() > 1:
            coors, dists = compute_mask_pixel_distances_with_coords(valley_mask.squeeze(0).squeeze(0))  # [n, 2]， [n, n]
            # print("valley_dist\n", dists)
            valley_ps = valley_p[0, 0, coors[:,0], coors[:,1]]
            valley_ps = torch.mean(valley_ps * (1-dists) , dim=1)    # [n,]
            valley_p[0, 0, coors[:, 0], coors[:, 1]] = valley_ps
        valley_p = valley_p / valley_p.max()

        return torch.cat([summit_p.squeeze(0), valley_p.squeeze(0)], dim=0)

    def initial_target(self, grad_intensity: torch.Tensor, pt_label: torch.Tensor, threshold: float = 0.1, fg_area=None, bg_area=None):
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
    
    def evolve_target(self, grad_intensity, target_mask, anti_advice, image, pt_label):
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
            result = self.initial_target(grad_intensity, pt_label, 0.1, fg_area=1-anti_advice.float(), bg_area=anti_advice)
            return result
        GI = grad_intensity
        image = (image - image.min())/(image.max() - image.min())
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
            # bg_mask = bg_area * summit_mask
            # # 生成空间均匀的seed
            # fg_mask = (GI == GI.max()) * fg_area
            # fg_seed_num = fg_area.sum() // 8
            # # fg_seed_num = fg_area.sum() // 8 - (fg_area*summit_mask).sum()
            # fg_mask = add_uniform_points_cuda(fg_area, fg_mask, int(fg_seed_num))

            bg_mask = torch.zeros_like(GI)
            bg_seed_num = bg_area.sum()/(fg_area.sum()+1)*fg_mask.sum()
            # bg_seed_num = bg_area.sum() // 8 - (bg_area*summit_mask).sum()
            # print("bg_seed_num")
            # print(bg_area.sum(), fg_area.sum(), fg_mask.sum(), bg_seed_num)
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

                    # # print(GI)
                    # print("result_")
                    # print(result_)
                    # print(fg_vwo)
                    # print(bg_vwo)
                    # print(fg_v)
                    # print(bg_v)

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

                fg_seed_num = int(0.1*max_num)
                fg_seed_num = fg_seed_num if fg_seed_num > 2 else 2
                fg_mask_new = add_uniform_points_v3(GI, (fg_area > 0.1) * (result_ < 0.), fg_mask>0.1, int(fg_seed_num), mode='fg')
                fg_mask_new = fg_mask_new.bool() * result

                bg_seed_num = int(0.1*min_num)
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
            for i in range(10):
                noise = torch.rand(GI.shape)
                GI = grad_intensity * (1-noise_ratio) + noise * noise_ratio

                fg_ratio = max(fg_ratio, 0.20)
                bg_ratio = max(bg_ratio, 0.20)

                local_max_num = 9 if max_num * fg_ratio < 9 else max_num * fg_ratio
                _, fg_vwo, _, fg_v = compute_weighted_mean_variance(GI, fg_mask > 0.1, int(local_max_num))

                local_min_num = 9 if min_num * bg_ratio < 9 else min_num * bg_ratio
                _, bg_vwo, _, bg_v = compute_weighted_mean_variance(GI, bg_mask > 0.1, int(local_min_num))
                result_total += fg_v/(fg_vwo+1e-8) -bg_v/(bg_vwo+1e-8)   #(H,W)
            result_total = keep_negative_by_top3_magnitude_levels(result_total, target_size=fg_area.sum())
            result = torch.where(result_total < 0., torch.ones_like(GI), torch.zeros_like(GI)).bool()
            result = filter_mask_by_points(result, target_mask, kernel_size=1).bool()

            fig = plt.figure(figsize=(25, 5))
            plt.subplot(1, 5, 1)
            plt.imshow(GI.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
            plt.subplot(1, 5, 2)
            plt.imshow(result_total, cmap='gray')
            plt.subplot(1, 5, 3)
            plt.imshow(bg_area, cmap='gray', vmax=1.0, vmin=0.0)
            plt.show(block=False)
            plt.subplot(1, 5, 4)
            plt.imshow(fg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
            plt.subplot(1, 5, 5)
            plt.imshow(bg_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
            a = input()

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

    def initial_target_v1(self, image: torch.Tensor, threshold: float = 0.1):
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
        summit_mask = local_norm_image == local_max2

        # 局部最小值
        local_min = min_pool2d(local_norm_image, (3,3), stride=1, padding=1)    # 考虑到背景的特性，使用更大的范围。
        valley_mask = local_norm_image == local_min

        fg_extremum_mask = torch.concatenate([summit_mask.squeeze(0) > threshold, valley_mask.squeeze(0) > threshold])
        bg_extremum_mask = torch.concatenate([valley_mask.squeeze(0) <= threshold, summit_mask.squeeze(0) <= threshold])

        fig = plt.figure(figsize=(35, 5))
        plt.subplot(1, 7, 1)
        plt.imshow(local_norm_image.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
        plt.subplot(1, 7, 2)
        plt.imshow(summit_mask.view(H, W), cmap='gray', vmax=1.0, vmin=0.0)
        plt.subplot(1, 7, 3)
        plt.imshow(valley_mask.view(H, W), cmap='gray')

        local_extremum_num = summit_mask.sum()  if summit_mask.sum() < valley_mask.sum() else valley_mask.sum()
        mask_channel_weight = torch.tensor([1.0, 0.5])
        fg_vwo, fg_v = compute_weighted_variance(image.squeeze(0).squeeze(0), fg_extremum_mask, mask_channel_weight, local_extremum_num)

        bg_vwo, bg_v = compute_weighted_variance(image.squeeze(0).squeeze(0), bg_extremum_mask, mask_channel_weight, local_extremum_num)

        plt.subplot(1, 7, 4)
        plt.imshow((bg_v - bg_vwo)/(bg_vwo+1e-8), cmap='gray', vmax=1.0, vmin=0.)
        plt.subplot(1, 7, 5)
        plt.imshow((fg_v - fg_vwo)/(fg_vwo+1e-8), cmap='gray', vmax=1.0, vmin=0.)
        plt.subplot(1, 7, 6)
        plt.imshow(bg_vwo, cmap='gray', vmax=1.0, vmin=0.)
        plt.subplot(1, 7, 7)
        plt.imshow(bg_v, cmap='gray', vmax=1.0, vmin=0.)
        plt.show(block=False)
        print(bg_vwo[10:22, 10:22])
        print(bg_v[10:22, 10:22])
        print(((fg_v - fg_vwo)/(fg_vwo+1e-8))[10:22, 10:22])
        result = torch.where((fg_v - fg_vwo)/fg_vwo > (bg_v - bg_vwo)/bg_vwo, torch.zeros_like(image.squeeze(0).squeeze(0)), torch.ones_like(image.squeeze(0).squeeze(0)))

        return result

    def generate_pseudo_labels(self, 
                             image: torch.Tensor,
                             additional_features: Optional[Dict[str, torch.Tensor]] = None,
                             confidence_threshold: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成最终的伪标签和置信度
        Args:
            image: 输入图像 [H, W]
            additional_features: 额外的特征字典，如卷积特征图，适用于像素的差别计算
            confidence_threshold: 置信度阈值，低于此阈值的概率将被设置为未标记
        Returns:
            pseudo_labels: 伪标签 [H, W]
            confidence: 置信度 [H, W]
        """
        target = self.initial_target(image, 0.1)

        # 形成种子点
        seeds = self._local_max_min_seed(image, 0.1)    #(2, H, W)

        # 执行随机游走推理
        probabilities = self.random_walk_inference(image, seeds.permute(1,2,0), additional_features)
        
        # 获取最大概率类别和置信度
        max_probs, predicted_labels = torch.max(probabilities, dim=2)
        
        # 应用置信度阈值
        low_confidence_mask = max_probs < confidence_threshold
        predicted_labels[low_confidence_mask] = 0  # 设置为未标记
        
        # print(predicted_labels, max_probs)
        # a = input()
        
        return predicted_labels, max_probs