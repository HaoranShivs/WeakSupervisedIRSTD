import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

from utils.sum_val_filter import min_pool2d

from typing import Tuple, Optional, Dict, List



def process_image_4_RW(image):
    """
    处理图像，找到种子
    Args:
        image: float(0,1), 输入的梯度强度，形状为(height, width)
    Returns:
        seeds: int, [H,W],1表示背景，2表示前景
    """
    image = image.unsqueeze(0).unsqueeze(0)
    # 局部最高值，由此来强化暗部，产生暗部的种子
    local_max = F.max_pool2d(image, (5,5), stride=1, padding=2)
    # 梯度强度➗局部最大值
    local_norm_image = image / (local_max + 1e-11)
    # 继而再找局部最大值
    local_max2 = F.max_pool2d(local_norm_image, (3,3 ), stride=1, padding=1)
    summit_mask = (local_norm_image == local_max2) * (image > 0.1)
    summit_mask = summit_mask.squeeze(0).squeeze(0)

    # 找局部最小值
    local_min = min_pool2d(image, (5,5), stride=1, padding=2)
    valley_mask = (image == local_min) * (image < 0.1)
    valley_mask = valley_mask.squeeze(0).squeeze(0)
    
    plt.figure(figsize=(25, 5))
    plt.subplot(151), plt.imshow(image.squeeze(0).squeeze(0), cmap='gray')
    plt.subplot(152), plt.imshow(local_max.squeeze(0).squeeze(0), cmap='gray')
    plt.subplot(153), plt.imshow(local_norm_image.squeeze(0).squeeze(0), cmap='gray')
    plt.subplot(154), plt.imshow(valley_mask, cmap='gray')
    plt.subplot(155), plt.imshow(summit_mask, cmap='gray')
    # plt.show()

    seeds = summit_mask.float() * 2 + valley_mask.float()

    return seeds


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
                 max_iterations: int = 100,
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
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.device = device
        
    def compute_transition_probabilities(self, 
                                       image: torch.Tensor,
                                       seeds: torch.Tensor,
                                       additional_features: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        计算转移概率矩阵
        
        Args:
            image: 输入图像 [H, W] 或 [H, W, C]
            seeds: 种子点标签 [H, W] (0表示未标记，>0表示类别标签)
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
            
        # 构建邻接关系（4邻域）
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
            additional_features
        )
        
        # 应用权重到邻接矩阵
        probabilities[row_indices, col_indices] = weights
        
        # 归一化行概率
        row_sum = torch.sum(probabilities, dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0  # 避免除零
        probabilities = probabilities / row_sum
        
        return probabilities
    
    def _build_adjacency_matrix(self, H: int, W: int) -> torch.Tensor:
        """构建邻接矩阵（4邻域）"""
        N = H * W
        adjacency = torch.zeros(N, N, device=self.device)
        
        # 4邻域：上、下、左、右
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
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
                            additional_features: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        计算边权重（联合概率）
        """
        # 空间距离权重
        spatial_weights = self._compute_spatial_weights(row_indices, col_indices, H, W)

        # 像素强度权重
        intensity_weights = self._compute_intensity_weights(row_indices, col_indices, flat_image)
        
        # 综合权重计算
        total_weights = torch.ones_like(spatial_weights)
        
        # 应用空间权重
        if self.spatial_weight > 0:
            total_weights *= torch.exp(-self.spatial_weight * spatial_weights)
            
        # 应用强度权重
        if self.intensity_weight > 0:
            total_weights *= torch.exp(-self.intensity_weight * intensity_weights)
            
        # 应用额外特征权重
        if additional_features and self.feature_weights:
            for feature_name, weight in self.feature_weights.items():
                if feature_name in additional_features:
                    feature_weights = self._compute_feature_weights(
                        row_indices, col_indices, 
                        additional_features[feature_name].reshape(-1, additional_features[feature_name].shape[-1])
                    )
                    total_weights *= torch.exp(-weight * feature_weights)
        
        return total_weights
    
    def _compute_spatial_weights(self, row_indices: torch.Tensor, 
                               col_indices: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """计算空间距离权重"""
        # 将线性索引转换为2D坐标
        row_i, row_j = row_indices // W, row_indices % W
        col_i, col_j = col_indices // W, col_indices % W
        
        # 计算欧几里得距离
        distances = torch.sqrt((row_i - col_i).float()**2 + (row_j - col_j).float()**2)
        return distances
    
    def _compute_intensity_weights(self, row_indices: torch.Tensor, 
                                 col_indices: torch.Tensor, 
                                 flat_image: torch.Tensor) -> torch.Tensor:
        """计算像素强度差异权重"""
        # sensity = (1/(flat_image[row_indices]+1e-4) + 1/(flat_image[col_indices]+1e-4))/2
        boundary = 0.1
        same_interval = (flat_image[row_indices] > boundary) ^ (flat_image[col_indices] > boundary) 
        diff = flat_image[row_indices] - flat_image[col_indices] + same_interval.float()
        return torch.norm(diff, dim=1)
    
    def _compute_feature_weights(self, row_indices: torch.Tensor, 
                               col_indices: torch.Tensor, 
                               flat_features: torch.Tensor) -> torch.Tensor:
        """计算额外特征差异权重"""
        diff = flat_features[row_indices] - flat_features[col_indices]
        return torch.norm(diff, dim=1)
    
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
    
    def random_walk_inference(self, image: torch.Tensor, seeds: torch.Tensor, additional_features) -> torch.Tensor:
        """
        正确的随机游走推理实现
        
        核心思想：
        1. 种子点的概率保持不变（吸收边界条件）
        2. 非种子点的概率根据邻居节点的概率进行更新
        3. 更新公式：p_i^(t+1) = Σ_j P_ij * p_j^t
           其中j是所有邻居节点（包括种子点和非种子点）
        """
        H, W = image.shape[:2]
        N = H * W
        
        # 获取唯一标签
        unique_labels = torch.unique(seeds)
        unique_labels = unique_labels[unique_labels > 0]
        num_classes = len(unique_labels)
        
        if num_classes == 0:
            raise ValueError("No seed labels provided!")
        
        # 标签映射
        label_mapping = {int(label): idx for idx, label in enumerate(unique_labels)}
        
        # 构建概率矩阵
        P = self.compute_transition_probabilities(image, seeds, additional_features)
        
        # 初始化概率矩阵 [N, num_classes]
        probabilities = torch.zeros(N, num_classes, device=self.device)
        seed_mask = (seeds.reshape(-1) > 0)
        
        # 设置种子点初始概率（吸收边界）
        for original_label, mapped_idx in label_mapping.items():
            mask = (seeds.reshape(-1) == original_label)
            probabilities[mask, mapped_idx] = 1.0
        
        # 分离转移概率矩阵
        # P_nn: 非种子点到非种子点的转移概率
        # P_ns: 非种子点到种子点的转移概率
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
        
        for iteration in range(self.max_iterations):
            # 只更新非种子点的概率
            # 新概率 = 从非种子点邻居传递的概率 + 从种子点邻居传递的概率
            seed_contrib = torch.matmul(P_ns, probabilities[seed_indices])  # 固定的种子贡献
            non_seed_contrib = torch.matmul(P_nn, current_probs[non_seed_indices])  # 来自非种子点的贡献
            
            # 更新非种子点概率
            current_probs[non_seed_indices] = seed_contrib + non_seed_contrib
            
            # 检查收敛性
            diff = torch.norm(current_probs - prev_probs)
            if diff < self.convergence_threshold:
                print(f"Converged at iteration {iteration}")
                break
                
            prev_probs = current_probs.clone()
        
        return current_probs.reshape(H, W, num_classes)

    def generate_pseudo_labels(self, 
                             image: torch.Tensor,
                             seeds: torch.Tensor,
                             additional_features: Optional[Dict[str, torch.Tensor]] = None,
                             confidence_threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成最终的伪标签和置信度
        
        Returns:
            pseudo_labels: 伪标签 [H, W]
            confidence: 置信度 [H, W]
        """
        # 执行随机游走推理
        probabilities = self.random_walk_inference(image, seeds, additional_features)
        
        # 获取最大概率类别和置信度
        max_probs, predicted_labels = torch.max(probabilities, dim=2)
        
        # 应用置信度阈值
        low_confidence_mask = max_probs < confidence_threshold
        predicted_labels[low_confidence_mask] = 0  # 设置为未标记
        
        return predicted_labels, max_probs