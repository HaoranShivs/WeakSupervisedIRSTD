import torch
import torch.nn.functional as F
import numpy as np


def iou_score(pred, target):
    smooth = 1e-11
    intersection = pred * target

    intersection_sum = np.sum(intersection, axis=(0,1))
    pred_sum = np.sum(pred, axis=(0,1))
    target_sum = np.sum(target, axis=(0,1))
    score = (intersection_sum + smooth) / \
            (pred_sum + target_sum - intersection_sum + smooth)

    score = np.mean(score)
    return score

def gaussian_kernel(size, sigma, kernel_dim=1): 
    coords = torch.arange(size, dtype=torch.float32)
    coords -= (size - 1) / 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()

    kernel_ = []
    for i in range(kernel_dim):  # 创建一个 [kernel_dim, size, size] 的张量
        kernel_.append(kernel_1d)

    kernel = torch.outer(*(kernel_)) if kernel_dim == 2 else kernel_1d
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel

def gaussian_blurring_2D(tensor, kernel_size, sigma):
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    kernel = gaussian_kernel(kernel_size, sigma, 2)
    result_mask = F.conv2d(tensor.float(), kernel, padding=kernel_size//2)
    return result_mask.squeeze(0).squeeze(0)

def extract_local_windows(tensor, window_size=5):
    """
    输入 tensor shape: (H, W)
    输出 windows shape: (H, W, window_size, window_size)
    """
    H, W = tensor.shape
    pad = window_size // 2
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    tensor_paded = F.pad(tensor, (pad, pad, pad, pad), mode="replicate")
    # 使用 unfold 提取局部块
    patches = F.unfold(tensor_paded, kernel_size=window_size)
    # reshape 成窗口形式
    _, c_kw2, n_patches = patches.shape
    patches = patches.transpose(1, 2).view(H, W, window_size, window_size)
    return patches

def mask_diameter(mask: torch.Tensor):
    """
    输入一个二值 mask (shape [H, W])，返回其区域内最远两点间的欧氏距离（即直径）
    
    Args:
        mask (Tensor): shape [H, W]，其中 1 表示目标区域，0 表示背景
    
    Returns:
        Tensor: 直径长度（标量）
    """
    assert mask.dim() == 2, "mask 必须是二维张量"
    
    # 获取设备信息
    device = mask.device
    
    # 获取 mask 中非零点的坐标
    coords = torch.nonzero(mask)  # shape [N, 2]
    
    if coords.shape[0] < 2:
        return torch.tensor(0.0, device=device)

    # 计算所有点之间的两两距离
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N, N, 2]
    diff = diff.float()
    dists = torch.hypot(diff[..., 0], diff[..., 1])   # 欧式距离

    # 取最大距离作为直径
    diameter = dists.max()

    return diameter

def compute_weighted_centroids(logits: torch.Tensor, edge_mask: torch.Tensor, topk_ratio=0.3):
    """
    计算高/低置信度区域的加权质心，考虑边缘附近像素的权重。
    
    Args:
        logits (Tensor): shape [H, W] 或 [C, H, W]
        edge_mask (Tensor): shape [H, W], 1 表示边缘区域
        topk_ratio (float): 取前多少比例的像素作为高/低置信度区域
    
    Returns:
        Tuple[Tensor, Tensor]: 高置信度质心 (y_high, x_high), 低置信度质心 (y_low, x_low)
    """
    assert edge_mask.shape == logits.shape[-2:], "edge_mask 和 logits 的空间尺寸必须一致"

    if logits.dim() == 3:
        logits = logits[0]  # 如果是多类，只取第一个类为例，你可以根据需要修改

    H, W = logits.shape
    device = logits.device

    # 创建坐标网格
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')

    # 边缘权重：1 + 边缘附近增强系数
    edge_mask_ = gaussian_blurring_2D(edge_mask, kernel_size=5, sigma=2.0)
    edge_weight = 1.0 + edge_mask_.float() * edge_mask

    # 归一化 logits（可选）
    logits_norm = (logits - logits.min()) / (logits.max() - logits.min() + 1e-8)

    # 取 top-k 像素作为高置信度区域
    num_pixels = H * W
    topk_num = max(1, int(num_pixels * topk_ratio))

    high_flat = logits_norm.view(-1)
    _, high_indices = torch.topk(high_flat, topk_num)
    high_mask = torch.zeros_like(high_flat).scatter_(0, high_indices, 1).reshape(H, W).bool()

    low_flat = (1 - logits_norm).view(-1)
    _, low_indices = torch.topk(low_flat, topk_num)
    low_mask = torch.zeros_like(low_flat).scatter_(0, low_indices, 1).reshape(H, W).bool()

    # 计算加权质心
    def weighted_centroid(mask, weight_map):
        weights = weight_map[mask]
        y_coords = y_grid[mask]
        x_coords = x_grid[mask]
        total_weight = weights.sum()
        cy = (y_coords * weights).sum() / total_weight.clamp(min=1e-8)
        cx = (x_coords * weights).sum() / total_weight.clamp(min=1e-8)
        return cy, cx

    centroid_high = weighted_centroid(high_mask, edge_weight)
    centroid_low = weighted_centroid(low_mask, edge_weight)

    return centroid_high, centroid_low

def farthest_point_sampling(mask, n_points):
    """
    在 mask 中选择 n_points 个二维空间上最远的点。
    
    参数:
        mask: 2D numpy array (H, W), dtype=bool 或 dtype=uint8 (0 or 1)
        n_points: 需要选择的点数
    
    返回:
        points: (n_points, 2) 的数组，每个点是 (y, x) 或 (row, col)
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)

    coords = np.argwhere(mask)

    if len(coords) == 0:
        raise ValueError("mask 中没有前景像素（值为1的点）。")

    if n_points > len(coords):
        raise ValueError(f"mask 中只有 {len(coords)} 个点，无法选出 {n_points} 个点。")

    selected_indices = []
    first_idx = np.random.randint(len(coords))
    selected_indices.append(first_idx)

    # 初始化每个点到当前选中点的最小距离
    min_distances = np.full(len(coords), np.inf)

    while len(selected_indices) < n_points:
        last_point = coords[selected_indices[-1]]
        # 更新所有点到最新选中点的距离
        dists = np.linalg.norm(coords - last_point, axis=1)
        min_distances = np.minimum(min_distances, dists)
        
        # 忽略已经选过的点
        min_distances[selected_indices] = -np.inf
        
        # 找出当前最远点（即 min_distances 最大的那个）
        farthest_idx = np.argmax(min_distances)
        selected_indices.append(farthest_idx)

    return coords[selected_indices]


def split_indices_by_mod(start, end, n, m):
    """
    生成指定范围内满足模 n 余 m 的索引列表，及其补集。

    参数:
        start (int): 起始数字（包含）
        end (int): 结束数字（包含）
        n (int): 模数，必须大于0
        m (int): 余数，应满足 0 <= m < n

    返回:
        tuple: (mod_list, complement_list)
               mod_list: 范围内满足 i % n == m 的数字
               complement_list: 范围内其余的数字（补集）
    """
    if n <= 0:
        raise ValueError("n 必须大于 0")
    if not (0 <= m < n):
        raise ValueError(f"m 必须满足 0 <= m < n，当前 m={m}, n={n}")

    full_range = range(start, end + 1)
    mod_list = [i for i in full_range if i % n == m]
    complement_list = [i for i in full_range if i % n != m]

    return mod_list, complement_list