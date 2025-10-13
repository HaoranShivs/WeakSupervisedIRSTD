import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

import math

# 全局缓存变量（初始为 None）
_GLOBAL_DIST_MATRIX_64 = None

def get_distance_matrix_64(device='cpu'):
    """
    返回 64x64 图像的全局像素距离矩阵 (4096, 4096)
    首次调用时计算并缓存，后续直接返回（自动迁移到指定 device）
    """
    global _GLOBAL_DIST_MATRIX_64

    if _GLOBAL_DIST_MATRIX_64 is None:
        # 只在 CPU 上计算一次（节省显存），后续按需迁移到 GPU
        # print("首次计算 64x64 距离矩阵...")
        H, W = 64, 64
        y = torch.arange(H, dtype=torch.float32)
        x = torch.arange(W, dtype=torch.float32)
        y_flat = y.repeat_interleave(W)  # (4096,)
        x_flat = x.repeat(H)             # (4096,)
        y_col = y_flat[:, None]  # (4096, 1)
        x_col = x_flat[:, None]
        y_row = y_flat[None, :]  # (1, 4096)
        x_row = x_flat[None, :]
        dist = torch.sqrt((y_col - y_row) ** 2 + (x_col - x_row) ** 2)
        _GLOBAL_DIST_MATRIX_64 = dist  # 保存在 CPU 上

    return _GLOBAL_DIST_MATRIX_64.to(device)


def dilate_mask(mask, d=2):
    kernel_size = 2 * d + 1
    weight = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    mask = mask.float()
    dilated = F.conv2d(mask, weight, padding=kernel_size // 2)
    dilated = (dilated > 0).float().squeeze()
    return dilated

def erode_mask(mask, d=2):
    kernel_size = 2 * d + 1
    weight = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    # 腐蚀等价于卷积后判断是否等于 kernel 中元素数量
    eroded = F.conv2d(mask.float(), weight, padding=kernel_size // 2)
    eroded = (eroded == kernel_size * kernel_size).float().squeeze()
    return eroded

def iou_score(pred, target):
    smooth = 1e-11
    intersection = pred * target

    intersection_sum = np.sum(intersection, axis=(0,1))
    pred_sum = np.sum(pred, axis=(0,1))
    target_sum = np.sum(target, axis=(0,1))
    score = (intersection_sum) / (pred_sum + target_sum - intersection_sum + smooth)

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

def compute_mask_pixel_distances_with_coords(mask):
    """
    计算 mask 中所有值为 1 的像素之间的距离，并返回坐标和距离矩阵。

    Args:
        mask (torch.Tensor): shape [H, W], dtype: int or bool

    Returns:
        coords (torch.Tensor): shape [n, 2], each row is (h, w)
        distances (torch.Tensor): shape [n, n], 每行是该点到其他点的距离
    """
    # 提取值为1的像素坐标
    coords = torch.nonzero(mask)  # shape [n, 2], (h, w)
    n = coords.shape[0]

    if n == 0:
        return coords, torch.empty(0, 0)
    if n == 1:
        return coords, torch.empty(1, 0)

    # 计算距离矩阵 [n, n]
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [n, n, 2]
    dists = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # 欧氏距离 [n, n]

    # # 去掉对角线，变成 [n, n-1]
    # eye = torch.eye(n, dtype=torch.bool, device=mask.device)
    # distances = dists[~eye].view(n, n - 1)
    
    # 归一化，使得距离在0-1之间
    dists = (dists - dists.min()) / (dists.max() - dists.min())

    return coords, dists


def min_positive_per_local_area(tensor, default=0.0):
    """
    在每个 local_area 中找到大于 0 的最小值。

    Args:
        tensor: shape [H, W, k, k]
        default: 当没有大于 0 的值时，返回的默认值

    Returns:
        result: shape [H, W], 每个位置是对应区域中 >0 的最小值
    """
    H, W, k1, k2 = tensor.shape
    flat = tensor.view(H, W, -1)  # [H, W, k1*k2]
    mask = flat > 0
    # 将非正数替换为 inf，避免影响 min
    inf_tensor = torch.full_like(flat, float('inf'))
    valid_values = torch.where(mask, flat, inf_tensor)
    min_vals = valid_values.min(dim=-1).values  # [H, W]
    # 替换 inf 为默认值
    result = torch.where(torch.isinf(min_vals), torch.tensor(default, dtype=min_vals.dtype, device=min_vals.device), min_vals)
    return result


def compute_local_extremes(image, mask, mode='max', local_size=3):
    """
    计算mask区域内像素的局部极值（最大值或最小值）
    
    Args:
        image: 形状为[H, W]的tensor，输入图像
        mask: 形状为[H, W]的tensor，二值掩码，True/1表示感兴趣区域
        mode: 字符串，'max'或'min'，决定计算最大值还是最小值
        local_size: 整数，pooling窗口大小，必须为奇数
    
    Returns:
        local_extremes: 形状为[H, W]的tensor，每个像素位置的局部极值
    """
    # assert image.dim() == 2, "image should be 2D tensor with shape [H, W]"
    # assert mask.dim() == 2, "mask should be 2D tensor with shape [H, W]"
    # assert image.shape == mask.shape, "image and mask should have same shape"
    # assert mode in ['max', 'min'], "mode should be 'max' or 'min'"
    # assert local_size % 2 == 1, "local_size should be odd number"
    
    H, W = image.shape
    padding = local_size // 2
    
    # 将image和mask扩展为适合卷积操作的形状 [1, 1, H, W]
    image_expanded = image.unsqueeze(0).unsqueeze(0)
    mask_expanded = mask.unsqueeze(0).unsqueeze(0).float()
    
    # 创建用于标记有效像素的扩展mask
    # 对mask进行same padding的卷积，统计每个窗口内有效像素数量
    local_extremes = torch.zeros_like(image)
    
    # 对每个像素位置计算局部极值
    for i in range(H):
        for j in range(W):
            # 如果当前像素不在mask区域内，跳过
            if not mask[i, j]:
                continue
                
            # 计算局部窗口的边界
            i_start = max(0, i - padding)
            i_end = min(H, i + padding + 1)
            j_start = max(0, j - padding)
            j_end = min(W, j + padding + 1)
            
            # 提取局部窗口
            local_image = image[i_start:i_end, j_start:j_end]
            local_mask = mask[i_start:i_end, j_start:j_end]
            
            # 只考虑mask为True的像素
            valid_pixels = local_image[local_mask]
            
            # 如果窗口内没有有效像素，使用原像素值
            if valid_pixels.numel() == 0:
                local_extremes[i, j] = image[i, j]
            else:
                if mode == 'max':
                    local_extremes[i, j] = valid_pixels.max()
                else:  # mode == 'min'
                    local_extremes[i, j] = valid_pixels.min()
    
    return local_extremes


# def compute_weighted_variance(logits: torch.Tensor, mask: torch.Tensor):
    """
    计算每个像素位置在 mask 区域内基于距离加权的方差（两种情况）：
    1. 不包含当前像素的加权方差
    2. 包含当前像素的加权方差

    Args:
        logits: (H, W) 的 Tensor
        mask: (H, W) 的二值 Tensor (0 or 1)

    Returns:
        var_wo: (H, W) 不包含当前像素的加权方差
        var_w:  (H, W) 包含当前像素的加权方差
    """
    assert logits.shape == mask.shape, "logits and mask must have the same shape"
    H, W = logits.shape

    # 预计算所有像素坐标
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=logits.device),
                                        torch.arange(W, device=logits.device),
                                        indexing='ij')  # (H, W)

    # 最大可能欧氏距离（从左上到右下）
    max_dist = math.sqrt((H - 1)**2 + (W - 1)**2)

    # 扩展坐标为 (H, W, 1, 1) 便于后续广播
    y_coords = y_coords.unsqueeze(-1).unsqueeze(-1)  # (H, W, 1, 1)
    x_coords = x_coords.unsqueeze(-1).unsqueeze(-1)

    # 获取 mask 中为 1 的所有位置的坐标
    mask_positions = mask.nonzero(as_tuple=False)  # (N, 2), each row is (i, j)
    N = mask_positions.shape[0]
    if N == 0:
        # 如果 mask 全为 0，返回全 0 方差
        var_wo = torch.zeros_like(logits)
        var_w = torch.zeros_like(logits)
        return var_wo, var_w

    # 提取 mask 区域内的 logits 值
    mask_logits_vals = logits[mask_positions[:, 0], mask_positions[:, 1]]  # (N,)

    # 构造 mask 区域的坐标张量 (N, 2) -> (1, 1, N, 2)
    mask_y = mask_positions[:, 0].view(1, 1, -1, 1)  # (1, 1, N, 1)
    mask_x = mask_positions[:, 1].view(1, 1, -1, 1)

    # 计算每个像素 (i,j) 到所有 mask 点的欧氏距离: (H, W, N, 1)
    dists = torch.sqrt((y_coords - mask_y)**2 + (x_coords - mask_x)**2)  # (H, W, N, 1)
    dists = dists.squeeze(-1)  # (H, W, N)

    # 计算权重: w = 1 - (d / max_dist), 距离超过 max_dist 的设为 0
    weights = 1 - dists / max_dist
    weights = torch.clamp(weights, min=0.0)  # (H, W, N)

    # 加权均值（不包含当前像素的情况）
    # 注意：mask_logits_vals 是 (N,)，需要扩展为 (1, 1, N)
    mask_logits_vals_exp = mask_logits_vals.unsqueeze(0).unsqueeze(0)  # (1, 1, N)

    # 加权和与权重和（用于均值）
    weighted_sum = torch.sum(weights * mask_logits_vals_exp, dim=-1)      # (H, W)
    total_weight = torch.sum(weights, dim=-1)                             # (H, W)

    # 防止除零
    safe_total_weight = torch.where(total_weight > 0, total_weight, torch.ones_like(total_weight))
    mean_wo = weighted_sum / safe_total_weight  # (H, W)

    # 计算方差：Var = sum(w * (x - mean)^2) / sum(w)
    # 先计算 (x - mean)^2，注意 mean 是 (H, W)，需要扩展为 (H, W, N)
    mean_exp = mean_wo.unsqueeze(-1)  # (H, W, 1)
    squared_diff = (mask_logits_vals_exp - mean_exp) ** 2  # (H, W, N)

    # 加权方差（不包含当前像素）
    var_numerator_wo = torch.sum(weights * squared_diff, dim=-1)
    var_wo = torch.where(total_weight > 0, var_numerator_wo / safe_total_weight, torch.zeros_like(var_numerator_wo))

    # =====================================================
    # 第二种情况：将当前像素也加入样本中
    # =====================================================

    # 当前像素的 logits 值: (H, W)
    current_logits = logits  # (H, W)

    # 将当前像素视为额外样本，添加到 mask 样本集中
    # 构造新的 logits 值: (H, W, N+1)
    current_logits_exp = current_logits.unsqueeze(-1)  # (H, W, 1)
    extended_logits = torch.cat([mask_logits_vals_exp.expand(H, W, N), current_logits_exp], dim=-1)  # (H, W, N+1)

    # 构造新的权重: 原 weights (H, W, N)，加上当前像素到自身的距离权重
    # 当前像素到 mask 中每个点的距离已经计算过，现在要计算当前像素到自身的权重？
    # 注意：当前像素到自己的距离为 0，权重为 1 - 0/max_dist = 1
    # 但我们还要计算当前像素到每个 mask 点的距离？不，我们只需要它自己的权重项
    # 实际上，我们只需添加一个新的权重维度：权重为 1.0（因为 d=0）

    # 新的权重张量: (H, W, N+1)
    current_weight = torch.ones((H, W, 1), device=logits.device)  # (H, W, 1)
    extended_weights = torch.cat([weights, current_weight], dim=-1)  # (H, W, N+1)

    # 计算新均值
    extended_weighted_sum = torch.sum(extended_weights * extended_logits, dim=-1)
    extended_total_weight = torch.sum(extended_weights, dim=-1)
    safe_extended_weight = torch.where(extended_total_weight > 0, extended_total_weight, torch.ones_like(extended_total_weight))
    mean_w = extended_weighted_sum / safe_extended_weight  # (H, W)

    # 计算新方差
    mean_w_exp = mean_w.unsqueeze(-1)  # (H, W, 1)
    squared_diff_w = (extended_logits - mean_w_exp) ** 2  # (H, W, N+1)
    var_numerator_w = torch.sum(extended_weights * squared_diff_w, dim=-1)
    var_w = torch.where(extended_total_weight > 0, var_numerator_w / safe_extended_weight, torch.zeros_like(var_numerator_w))

    return var_wo, var_w


def compute_weighted_variance_v1(
    logits: torch.Tensor,
    mask: torch.Tensor,
    channel_weights: torch.Tensor,
    top_k: int = None,
    thre: float = 0.0
):
    """
    计算每个像素位置在 multi-channel mask 区域内基于距离和通道权重联合加权的方差。

    Args:
        logits: (H, W) 或 (C, H, W) 的 Tensor。若为 (C,H,W)，则按通道取值；若为 (H,W)，则共享
        mask: (C, H, W) 的二值 Tensor
        channel_weights: (C,) 的 Tensor，表示每个通道的权重
        top_k: int, 使用最近的 top_k 个点（跨通道）
        thre: float, 当所有通道总共只有一个有效点时，用于 fallback 的虚拟点取值

    Returns:
        var_wo: (H, W) 不包含当前像素的加权方差
        var_w:  (H, W) 包含当前像素的加权方差
    """
    assert mask.ndim == 3, "mask must be (C, H, W)"
    C, H, W = mask.shape

    if logits.shape == (H, W):
        # 扩展 logits 到 (C, H, W)
        logits_exp = logits.unsqueeze(0).expand(C, H, W)
    elif logits.shape == (C, H, W):
        logits_exp = logits
    else:
        raise ValueError(f"logits must be (H, W) or (C, H, W), got {logits.shape}")

    assert channel_weights.shape == (C,), f"channel_weights must be (C,), got {channel_weights.shape}"

    # 预计算坐标
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=mask.device),
                                        torch.arange(W, device=mask.device),
                                        indexing='ij')
    y_coords = y_coords.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    x_coords = x_coords.unsqueeze(0).unsqueeze(0)

    # 获取所有 mask 为 True 的点：(N, 3) -> [c, y, x]
    indices = mask.nonzero(as_tuple=False)  # (N, 3)
    N = indices.shape[0]

    if N == 0:
        var_wo = torch.zeros(H, W, device=mask.device)
        var_w = torch.zeros(H, W, device=mask.device)
        return var_wo, var_w

    # 提取每个有效点的 c, y, x
    c_idx = indices[:, 0]  # (N,)
    y_idx = indices[:, 1]  # (N,)
    x_idx = indices[:, 2]  # (N,)

    # 获取这些点的 logits 值
    point_logits = logits_exp[c_idx, y_idx, x_idx]  # (N,)

    # 获取每个点的通道权重
    point_channel_weights = channel_weights[c_idx]  # (N,)

    # 计算每个当前像素 (i,j) 到每个 mask 点的距离
    # 当前坐标: (1, 1, H, W)
    # 点坐标: (N,) -> (N, 1, 1)
    y_idx = y_idx.view(N, 1, 1)
    x_idx = x_idx.view(N, 1, 1)
    dists = torch.sqrt((y_coords - y_idx)**2 + (x_coords - x_idx)**2)  # (N, H, W)
    dists = dists.squeeze(0)
    dists = dists.permute(1, 2, 0)  # (H, W, N)

    # === 特殊情况：仅有一个有效点 ===
    if N == 1:
        # 添加一个虚拟点：距离相同，logits = thre，channel_weight = 该点的权重
        dists_used = torch.cat([dists, dists], dim=-1)  # (H, W, 2)

        # 归一化
        d_min = dists_used.min(dim=-1, keepdim=True).values
        d_max = dists_used.max(dim=-1, keepdim=True).values
        eps = 1e-8
        normalized_d = (dists_used - d_min) / (d_max - d_min + eps)
        dist_weights = 1 - normalized_d  # (H, W, 2)
        dist_weights = torch.clamp(dist_weights, min=0.0)

        # 综合权重 = 距离权重 × 通道权重（广播）
        # 原始点权重
        orig_cw = point_channel_weights[0]  # scalar
        cw_weights = torch.tensor([orig_cw, orig_cw], device=mask.device)  # (2,)
        cw_weights = cw_weights.view(1, 1, 2).expand(H, W, 2)  # (H, W, 2)
        weights = dist_weights * cw_weights  # (H, W, 2)

        # logits 值
        val0 = point_logits[0]
        logits_vals = torch.full((H, W, 2), thre, device=mask.device)
        logits_vals[:, :, 0] = val0  # (H, W, 2)

    else:
        # 正常情况：N >= 2
        k = min(top_k, N) if top_k is not None else N

        # 归一化距离（每个像素独立）
        d_min = dists.min(dim=-1, keepdim=True).values    # (H, W, 1)
        d_max = dists.max(dim=-1, keepdim=True).values
        eps = 1e-8
        normalized_d = (dists - d_min) / (d_max - d_min + eps)
        dist_weights = 1 - normalized_d  # (H, W, N)
        dist_weights = torch.clamp(dist_weights, min=0.0)

        # 综合权重：距离权重 × 通道权重
        # point_channel_weights: (N,) -> (1, 1, N)
        cw_exp = point_channel_weights.view(1, 1, -1).expand(H, W, N)
        weights = dist_weights * cw_exp  # (H, W, N)

        # 可选：只取 top_k
        if k < N:
            topk_weights, topk_indices = torch.topk(weights, k, dim=-1)  # (H, W, k)
            logits_vals = point_logits.unsqueeze(0).unsqueeze(0).expand(H, W, N)
            logits_vals = torch.gather(logits_vals, dim=-1, index=topk_indices)
            weights = topk_weights
            dists_used = torch.gather(dists, dim=-1, index=topk_indices)
        else:
            logits_vals = point_logits.unsqueeze(0).unsqueeze(0).expand(H, W, N)
            dists_used = dists

    # === 情况一：不包含当前像素的加权方差 ===
    weighted_sum = torch.sum(weights * logits_vals, dim=-1)  # (H, W)
    total_weight = torch.sum(weights, dim=-1)                # (H, W)
    safe_weight = torch.where(total_weight > 0, total_weight, torch.ones_like(total_weight))
    mean_wo = weighted_sum / safe_weight  # (H, W)

    mean_exp = mean_wo.unsqueeze(-1)  # (H, W, 1)
    squared_diff = (logits_vals - mean_exp) ** 2
    var_numerator_wo = torch.sum(weights * squared_diff, dim=-1)
    var_wo = torch.where(total_weight > 0, var_numerator_wo / safe_weight, torch.zeros_like(var_numerator_wo))

    # === 情况二：包含当前像素 ===
    current_logits = logits if logits.ndim == 2 else logits.mean(dim=0)  # (H, W)
    current_logits = current_logits.unsqueeze(-1)  # (H, W, 1)

    # 计算当前像素到各 mask 点的距离（用于权重）
    if N == 1:
        # 使用与原点相同的 dists（已扩展）
        current_dists = dists  # (H, W, 1)
        d_min_cur = current_dists
        d_max_cur = current_dists
        normalized_cur_d = (torch.zeros_like(d_min_cur) - d_min_cur) / (d_max_cur - d_min_cur + eps)
        dist_weight_cur = 1 - normalized_cur_d
        # 通道权重：与原点相同
        cw_cur = point_channel_weights[0].view(1, 1, 1).expand(H, W, 1)
        current_weight_val = dist_weight_cur * cw_cur
    else:
        # 使用 dists_used (H, W, k or N)
        d_min_cur = dists_used.min(dim=-1, keepdim=True).values
        d_max_cur = dists_used.max(dim=-1, keepdim=True).values
        current_dists = torch.zeros_like(d_min_cur)
        normalized_cur_d = (current_dists - d_min_cur) / (d_max_cur - d_min_cur + eps)
        dist_weight_cur = 1 - normalized_cur_d
        dist_weight_cur = torch.clamp(dist_weight_cur, min=0.0)
        # 通道权重：暂用平均或最大？这里我们不区分，只用距离 + 统一通道权重逻辑
        # 实际上，当前像素不属于任何通道，我们只考虑其距离权重 × ？？
        # 更合理：当前像素不带通道权重，只保留距离权重部分
        # 所以我们只用 dist_weight_cur，不再乘 channel weight
        current_weight_val = dist_weight_cur  # (H, W, 1)

    # 拼接
    extended_logits = torch.cat([logits_vals, current_logits], dim=-1)  # (H, W, *)
    extended_weights = torch.cat([weights, current_weight_val], dim=-1)  # (H, W, *)

    # 新均值
    ext_sum = torch.sum(extended_weights * extended_logits, dim=-1)
    ext_weight_sum = torch.sum(extended_weights, dim=-1)
    safe_ext_weight = torch.where(ext_weight_sum > 0, ext_weight_sum, torch.ones_like(ext_weight_sum))
    mean_w = ext_sum / safe_ext_weight

    # 新方差
    mean_w_exp = mean_w.unsqueeze(-1)
    squared_diff_w = (extended_logits - mean_w_exp) ** 2
    var_numerator_w = torch.sum(extended_weights * squared_diff_w, dim=-1)
    var_w = torch.where(ext_weight_sum > 0, var_numerator_w / safe_ext_weight, torch.zeros_like(var_numerator_w))

    return var_wo, var_w

def compute_weighted_variance_v2(
    logits: torch.Tensor,
    mask: torch.Tensor,
    top_k: int = None,
    thre: float = 0.0
):
    """
    计算每个像素位置在 mask 区域内基于距离加权的方差（两种情况）：
    1. 不包含当前像素的加权方差
    2. 包含当前像素的加权方差
    使用 mask 中距离最近的 top_k 个点参与计算。
    距离权重归一化基于：当前像素到这些 mask 点的 min 和 max 距离。

    特殊处理：当 mask 中只有一个点时，添加一个虚拟点（值=thre，权重同原点）用于计算方差。

    Args:
        logits: (H, W) 的 Tensor
        mask: (H, W) 的二值 Tensor (0 or 1)
        top_k: int, 使用 mask 中距离最近的 top_k 个点；若为 None，则使用所有 mask 点
        thre: float, 当 mask 只有一个点时，添加的虚拟点的取值（用于 fallback）

    Returns:
        var_wo: (H, W) 不包含当前像素的加权方差
        var_w:  (H, W) 包含当前像素的加权方差
    """
    assert logits.shape == mask.shape, "logits and mask must have the same shape"
    H, W = logits.shape

    # 预计算所有像素坐标
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=logits.device),
                                        torch.arange(W, device=logits.device),
                                        indexing='ij')  # (H, W)
    y_coords = y_coords.unsqueeze(-1).unsqueeze(-1)  # 扩展坐标为 (H, W, 1, 1) 便于广播
    x_coords = x_coords.unsqueeze(-1).unsqueeze(-1)

    # 获取 mask 中为 1 的所有位置的坐标
    mask_positions = mask.nonzero(as_tuple=False)  # (N, 2)
    N = mask_positions.shape[0]

    # 正常情况：N >= 4
    k = min(top_k, N) if top_k is not None else N
    use_all = k >= N

    # 提取 mask 区域内的 logits 值
    mask_logits_vals = logits[mask_positions[:, 0], mask_positions[:, 1]]  # (N,)

    # 构造 mask 坐标张量
    mask_y = mask_positions[:, 0].view(1, 1, -1, 1)  # (1, 1, N, 1)
    mask_x = mask_positions[:, 1].view(1, 1, -1, 1)

    # 计算距离: (H, W, N)
    dists = torch.sqrt((y_coords - mask_y)**2 + (x_coords - mask_x)**2).squeeze(-1)  # (H, W, N)

    if not use_all:
        # 取最近的 k 个点
        topk_dists, topk_indices = torch.topk(dists, k, dim=-1, largest=False)  # (H, W, k)
        mask_logits_vals_exp = mask_logits_vals.unsqueeze(0).unsqueeze(0).expand(H, W, N)
        topk_logits = torch.gather(mask_logits_vals_exp, dim=-1, index=topk_indices)  # (H, W, k)
        dists_used = topk_dists
    else:
        topk_logits = mask_logits_vals.unsqueeze(0).unsqueeze(0).expand(H, W, N)  # (H, W, N)
        dists_used = dists  # (H, W, k)

        # 动态归一化
        d_min = dists_used.min(dim=-1, keepdim=True).values
        d_max = dists_used.max(dim=-1, keepdim=True).values
        eps = 1e-8
        normalized_d = (dists_used - d_min) / (d_max - d_min + eps)
        weights = 1 - normalized_d
        weights = torch.clamp(weights, min=0.0)  # (H, W, k)

    # === 情况一：不包含当前像素的加权方差 ===
    weighted_sum = torch.sum(weights * topk_logits, dim=-1)      # (H, W)
    total_weight = torch.sum(weights, dim=-1)                    # (H, W)
    safe_total_weight = torch.where(total_weight > 0, total_weight, torch.ones_like(total_weight))
    mean_wo = weighted_sum / safe_total_weight  # (H, W)

    mean_exp = mean_wo.unsqueeze(-1)  # (H, W, 1)
    squared_diff = (topk_logits - mean_exp) ** 2  # (H, W, *)
    var_numerator_wo = torch.sum(weights * squared_diff, dim=-1)
    var_wo = torch.where(total_weight > 0, var_numerator_wo / safe_total_weight, torch.zeros_like(var_numerator_wo))

    # === 情况二：包含当前像素 ===
    current_logits = logits.unsqueeze(-1)  # (H, W, 1)

    # 计算当前像素到 mask 区域的距离（用于 fallback 权重）
    if N == 1:
        # 当前像素到唯一 mask 点的距离
        y0, x0 = mask_positions[0]
        current_dists = torch.sqrt((y_coords.squeeze() - y0)**2 + (x_coords.squeeze() - x0)**2)  # (H, W)
        current_dists = current_dists.unsqueeze(-1)  # (H, W, 1)
        # 归一化时使用与原点相同的 d_min/d_max（即自身距离）
        d_min_cur = current_dists
        d_max_cur = current_dists
        normalized_cur_d = (current_dists - d_min_cur) / (d_max_cur - d_min_cur + eps)  # 0
        current_weight_val = 1 - normalized_cur_d  # (H, W, 1)
    else:
        # 正常情况：使用当前像素到所选 mask 点的距离
        current_dists = dists_used  # 已经是 (H, W, k) 或 (H, W, 2)
        d_min_cur = d_min
        d_max_cur = d_max
        eps = 1e-8
        normalized_cur_d = (torch.zeros_like(d_min_cur) - d_min_cur) / (d_max_cur - d_min_cur + eps)
        current_weight_val = 1 - normalized_cur_d
        current_weight_val = torch.clamp(current_weight_val, min=0.0)

    # 拼接当前像素
    extended_logits = torch.cat([topk_logits, current_logits], dim=-1)  # (H, W, k+1 或 3)
    extended_weights = torch.cat([weights, current_weight_val], dim=-1)  # (H, W, k+1 或 3)

    # 新均值
    extended_weighted_sum = torch.sum(extended_weights * extended_logits, dim=-1)
    extended_total_weight = torch.sum(extended_weights, dim=-1)
    safe_extended_weight = torch.where(extended_total_weight > 0, extended_total_weight, torch.ones_like(extended_total_weight))
    mean_w = extended_weighted_sum / safe_extended_weight  # (H, W)
    # print(mean_w)

    # 新方差
    mean_w_exp = mean_w.unsqueeze(-1)
    squared_diff_w = (extended_logits - mean_w_exp) ** 2
    var_numerator_w = torch.sum(extended_weights * squared_diff_w, dim=-1)
    var_w = torch.where(extended_total_weight > 0, var_numerator_w / safe_extended_weight, torch.zeros_like(var_numerator_w))

    return var_wo, var_w

def random_select_from_mask(mask, num):
    """
    从 mask 中值为 1 的位置随机选择 num 个，返回新的 mask。
    
    参数:
        mask (torch.Tensor): 值为 0 或 1 的二值 mask，形状为 [H, W] 或 [1, H, W] 等
        num (int): 要选择的像素数量
    
    返回:
        torch.Tensor: 新的 mask，只保留随机选中的 num 个像素（值为 1）
    """
    # 确保 mask 是 bool 或 0/1 的整数类型
    if mask.dtype != torch.bool:
        mask_bool = mask.bool()
    else:
        mask_bool = mask

    # 获取值为 1 的像素的索引
    indices = torch.nonzero(mask_bool, as_tuple=False)  # 形状为 [N, dim]

    # 如果 1 的数量少于或等于 num，直接返回原 mask（或全选）
    if indices.size(0) <= num:
        return mask.clone()

    # 随机打乱并选择 num 个索引
    perm = torch.randperm(indices.size(0), device=mask.device)
    selected_indices = indices[perm[:num]]

    # 创建输出 mask，初始化为 0
    out_mask = torch.zeros_like(mask, dtype=torch.bool)

    # 将选中的位置设为 True
    if mask.dim() == 2:
        out_mask[selected_indices[:, 0], selected_indices[:, 1]] = True
    elif mask.dim() == 3:
        out_mask[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]] = True
    else:
        raise ValueError("只支持 2D 或 3D 的 mask")

    return out_mask


def random_select_from_prob_mask(prob_mask, num, replacement=False):
    """
    根据 float 类型的概率 mask，按概率权重随机选择 num 个像素，返回二值 mask。

    参数:
        prob_mask (torch.Tensor): float 类型，形状为 [H, W] 或 [C, H, W]（C=1），值在 [0,1] 表示选中概率
        num (int): 要选择的像素数量
        replacement (bool): 是否允许重复采样（通常 False）

    返回:
        torch.Tensor: 二值 mask，long 类型，选中的位置为 1，其余为 0
    """
    # 保存原始设备和形状
    device = prob_mask.device
    shape = prob_mask.shape

    # 展平为一维便于采样
    if prob_mask.dim() == 3:
        if prob_mask.size(0) != 1:
            raise ValueError("3D mask 必须是 [1, H, W] 形式")
        flat_probs = prob_mask.view(-1)  # [C*H*W]
    elif prob_mask.dim() == 2:
        flat_probs = prob_mask.reshape(-1)  # [H*W]
    else:
        raise ValueError("只支持 2D 或 3D 的 prob_mask")

    # 检查 num 是否合法
    total_elements = flat_probs.numel()
    if num > total_elements and not replacement:
        raise ValueError(f"无法在不放回的情况下选择 {num} 个元素（总共只有 {total_elements} 个）")

    # 使用 multinomial 按概率采样（支持零概率）
    try:
        indices = torch.multinomial(flat_probs, num, replacement=replacement)
    except RuntimeError as e:
        if "multinomial" in str(e) and not replacement:
            # 可能是因为非零数少于 num 且不放回
            # 我们可以先归一化非零部分，或报更友好错误
            nonzero_count = (flat_probs > 0).sum().item()
            if nonzero_count < num:
                raise ValueError(
                    f"非零概率的位置只有 {nonzero_count} 个，"
                    f"但要求选择 {num} 个像素，请确保 num 不超过非零位置数，"
                    f"或使用 replacement=True"
                ) from e
            else:
                raise e
        else:
            raise e

    # 创建输出 mask
    out_mask = torch.zeros_like(flat_probs)
    out_mask[indices] = 1

    # 恢复原始形状
    out_mask = out_mask.reshape(shape)

    return out_mask

# 辅助函数：用于 fallback 的独立均匀采样函数
def select_uniform_logits_pixels_v2(logits, mask_a, mask_b, num, num_bins=20):
    """简化版：在 mask_a \ mask_b 中按值域分桶均匀采样"""
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    H, W = logits.shape

    if mask_a.dtype == torch.int:
        mask_a = mask_a.bool()
    if mask_b.dtype == torch.int:
        mask_b = mask_b.bool()

    candidate_mask = mask_a & (~mask_b)
    candidate_indices = candidate_mask.nonzero(as_tuple=False)
    N = candidate_indices.size(0)
    if N == 0 or num == 0:
        return mask_b.clone()
    if num > N:
        num = N

    candidate_logits = logits[candidate_mask]
    min_val = candidate_logits.min().item()
    max_val = candidate_logits.max().item()

    if abs(max_val - min_val) < 1e-6:
        idx = torch.randperm(N)[:num]
        selected = candidate_indices[idx]
    else:
        bin_edges = torch.linspace(min_val, max_val, num + 1, device=logits.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        selected = []
        for center in bin_centers:
            dist = torch.abs(candidate_logits - center)
            best = torch.argmin(dist)
            selected.append(candidate_indices[best])
            # 移除已选
            if len(selected) < num:
                keep = torch.ones(candidate_logits.shape, dtype=torch.bool, device=logits.device)
                keep[best] = False
                candidate_logits = candidate_logits[keep]
                candidate_indices = candidate_indices[keep]
        selected = torch.stack(selected)
    mask_c = mask_b.clone()
    for h, w in selected:
        mask_c[h, w] = True
    return mask_c

def select_complementary_pixels(logits, mask_a, mask_b, num, num_bins=20):
    """
    选择 num 个新像素，使得它们与 mask_b 中的像素合并后，
    对应的 logits 值在数值上尽可能均匀分布。

    参数：
        logits: [H, W] 或 [1, H, W]
        mask_a: bool [H, W], 候选区域
        mask_b: bool [H, W], mask_b ⊆ mask_a
        num: int, 要新增的像素数
        num_bins: int, 用于统计分布的桶数

    返回：
        mask_c: bool [H, W], mask_b + 新选的 num 个像素
    """
    if mask_a.dtype == torch.int:
        mask_a = mask_a.bool()
    if mask_b.dtype == torch.int:
        mask_b = mask_b.bool()

    if logits.dim() == 3:
        logits = logits.squeeze(0)
    H, W = logits.shape

    # 1. 获取 mask_b 中的 logits 值（已有样本）
    existing_mask = mask_b
    existing_logits = logits[existing_mask]  # [M]
    M = existing_logits.numel()

    # 如果没有现有样本，退化为在候选中均匀采样
    if M == 0:
        return select_uniform_logits_pixels_v2(logits, mask_a, mask_b, num, num_bins)

    # 2. 定义全局值域：基于 mask_a 中的所有值
    universe_mask = mask_a
    global_logits = logits[universe_mask]
    
    # 👉 检查 global_logits 是否为空
    if global_logits.numel() == 0:
        # mask_a 中没有有效像素，无法选择，直接返回 mask_b
        print("Warning: mask_a has no active pixels. Returning mask_b.")
        return mask_b.clone()

    min_val = global_logits.min().item()
    max_val = global_logits.max().item()

    if abs(max_val - min_val) < 1e-6:
        # 所有值几乎相等，随便选
        candidate_mask = mask_a & (~mask_b)
        candidate_indices = candidate_mask.nonzero()
        N = candidate_indices.size(0)
        if N == 0:
            return mask_b.clone()
        num = min(num, N)
        selected_idx = torch.randperm(N)[:num]
        selected_coords = candidate_indices[selected_idx]
    else:
        # 3. 划分 bins
        bin_edges = torch.linspace(min_val, max_val, steps=num_bins + 1, device=logits.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # [num_bins]

        # 4. 统计 mask_b 在各 bin 中的数量
        bin_indices_existing = torch.bucketize(existing_logits, bin_edges, right=False) - 1
        bin_indices_existing = torch.clamp(bin_indices_existing, 0, num_bins - 1)
        bin_counts = torch.bincount(bin_indices_existing, minlength=num_bins).float()  # [num_bins]

        # 5. 按“谁最缺”排序：优先补充样本少的桶
        # 我们希望每个桶最终有 roughly (M + num) / num_bins 个样本
        desired_per_bin = (M + num) / num_bins
        deficit = desired_per_bin - bin_counts  # 缺多少
        _, sorted_deficit_bins = torch.sort(deficit, descending=True)  # 从缺得最多到少

        # 6. 候选区域：mask_a 且不在 mask_b
        candidate_mask = mask_a & (~mask_b)
        candidate_indices = candidate_mask.nonzero(as_tuple=False)  # [N, 2]
        N = candidate_indices.size(0)

        if N == 0:
            print("No candidates available.")
            return mask_b.clone()

        candidate_logits_vals = logits[candidate_mask]  # [N]

        # 7. 为每个高优先级桶选点
        selected_coords = []

        # 遍历所有桶（按缺损排序），直到选够 num 个
        for bin_idx in sorted_deficit_bins:
            if len(selected_coords) >= num:
                break

            # 找候选中落在这个桶内的像素
            in_bin = (candidate_logits_vals >= bin_edges[bin_idx]) & \
                     (candidate_logits_vals < bin_edges[bin_idx + 1])

            if not in_bin.any():
                continue

            # 在该桶内，选最接近桶中心的像素
            distances = torch.abs(candidate_logits_vals[in_bin] - bin_centers[bin_idx])
            best_local = torch.argmin(distances)
            # 映射回原始候选索引
            candidate_in_bin = candidate_indices[in_bin]
            coord = candidate_in_bin[best_local]

            # 添加并从候选中移除（避免重复选）
            selected_coords.append(coord)

            # 从候选中移除这个点（更新 candidate_indices 和 candidate_logits_vals）
            mask = (candidate_indices != coord).all(dim=1)
            candidate_indices = candidate_indices[mask]
            candidate_logits_vals = candidate_logits_vals[mask[::]]  # 注意：mask 长度=N，candidate_logits_vals 长度=N

            # 提前终止
            if len(selected_coords) == num:
                break

        # 如果还没选够，用 fallback 补齐
        if len(selected_coords) < num:
            remaining_num = num - len(selected_coords)

            # 构造当前已包含的 mask_b + 已选点
            temp_mask_b = mask_b.clone()
            if selected_coords:
                selected_tensor = torch.stack(selected_coords)
                temp_mask_b[selected_tensor[:, 0], selected_tensor[:, 1]] = True

            # 调用 fallback 函数（确保不重复选）
            fallback_mask = select_uniform_logits_pixels_v2(
                logits, mask_a, temp_mask_b, remaining_num, num_bins=num_bins
            )

            # 从 fallback_mask 中提取真正新增的点（不在原来的 mask_b 和 selected_coords 中）
            combined_so_far = mask_b.clone()
            if selected_coords:
                coords_tensor = torch.stack(selected_coords)
                combined_so_far[coords_tensor[:, 0], coords_tensor[:, 1]] = True

            # 找出 fallback_mask 中比 combined_so_far 多出的部分
            new_points_mask = fallback_mask & (~combined_so_far)
            new_points = new_points_mask.nonzero(as_tuple=False)

            # 添加最多 remaining_num 个
            add_count = min(remaining_num, new_points.size(0))
            if add_count > 0:
                selected_coords.extend([new_points[i] for i in range(add_count)])

        selected_coords = torch.stack(selected_coords) if selected_coords else torch.empty((0, 2), dtype=torch.long)

    # 8. 构造最终 mask
    mask_c = mask_b.clone()
    for i in range(selected_coords.size(0)):
        h, w = selected_coords[i]
        mask_c[h, w] = True

    return mask_c

def get_connected_mask_long_side(mask):
    """
    输入一个形状为 [H, W] 的 PyTorch tensor，其中值为 1 的像素构成一个连通区域。
    返回该连通区域外接矩形的长边长度。

    参数:
        mask (torch.Tensor): shape [H, W], dtype=torch.uint8 or bool or float, 值为 0 或 1

    返回:
        int: 竖边长度
        int: 横边长度
    """
    assert mask.dim() == 2, "mask must be 2D (H, W)"

    # 将 mask 转为 bool 类型
    mask = mask.bool()

    # 获取所有值为 1 的像素的坐标
    coords = torch.nonzero(mask, as_tuple=False)  # 形状为 [N, 2], 每行是 (y, x)

    if coords.numel() == 0:
        return 0  # 没有前景像素

    # 提取 x 和 y 坐标
    y_coords = coords[:, 0]
    x_coords = coords[:, 1]

    # 计算边界框
    y_min = y_coords.min().item()
    y_max = y_coords.max().item()
    x_min = x_coords.min().item()
    x_max = x_coords.max().item()

    # 计算矩形的宽度和高度
    height = y_max - y_min + 1
    width = x_max - x_min + 1

    # 返回长边
    return height, width

def compute_weighted_mean_variance(logits: torch.Tensor, mask: torch.Tensor, top_k: int = None):
    """
    计算每个像素在 mask 区域内基于距离加权的方差：
    - var_wo: 不包含当前像素（即使它在 mask 中）
    - var_w:  包含当前像素
    使用 mask 中最近的 top_k 个点，支持 N < top_k 的 padding。
    权重使用指数衰减，避免远点权重为0。
    """
    assert logits.shape == mask.shape
    H, W = logits.shape
    device = logits.device

    # 坐标网格 (H, W, 2)
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device),
                                        torch.arange(W, device=device), indexing='ij')
    coords = torch.stack([y_coords, x_coords], dim=-1).float()  # (H, W, 2)

    # 所有 mask == 1 的位置
    mask_positions = mask.nonzero(as_tuple=False)  # (N, 2)
    N = len(mask_positions)
    if N == 0:
        zero = torch.zeros_like(logits)
        return zero, zero, zero, zero

    mask_coords = mask_positions.float()  # (N, 2)
    mask_logits_vals = logits[mask_positions[:, 0], mask_positions[:, 1]]  # (N,)

    # 当前像素是否在 mask 中？(H, W)
    self_in_mask = mask.bool()  # (H, W)

    # 所有点到 mask 点的距离: (H, W, N)
    all_dists = torch.cdist(coords.view(H*W, 2), mask_coords.unsqueeze(0)).view(H, W, N)

    # === 正确处理 top_k，即使 k > N ===
    k_pad = top_k if top_k is not None else N
    k_use = min(k_pad, N) if top_k is not None else N

    # 1. 先对 all_dists 做 topk，限制 k_use <= N
    topk_dists_full, topk_indices_full = torch.topk(all_dists, k_use, dim=-1, largest=False)  # (H, W, k_use)

    # 2. Gather 对应的 logits
    # mask_logits_vals: (N,)
    topk_logits = torch.gather(
        mask_logits_vals.unsqueeze(0).unsqueeze(0).expand(H, W, N),
        dim=-1,
        index=topk_indices_full
    )  # (H, W, k_use)

    # 3. 如果需要 padding 到 k_pad，才进行 pad
    if k_use < k_pad:
        pad_size = k_pad - k_use
        # Pad distances with inf
        topk_dists = torch.cat([topk_dists_full, topk_dists_full.new_full((H, W, pad_size), float('inf'))], dim=-1)  # (H, W, k_pad)
        # Pad logits with 0 (won't affect due to weight=0)
        topk_logits = torch.cat([topk_logits, topk_logits.new_zeros(H, W, pad_size)], dim=-1)
        # Pad indices: 用 -1 表示无效，或者随便填（不重要）
        topk_indices = torch.cat([topk_indices_full, topk_indices_full.new_full((H, W, pad_size), -1)], dim=-1)
        valid_mask = torch.cat([
            torch.ones_like(topk_indices_full, dtype=torch.bool),
            torch.zeros(H, W, pad_size, dtype=torch.bool, device=device)
        ], dim=-1)
    else:
        topk_dists = topk_dists_full  # (H, W, k_pad)
        topk_indices = topk_indices_full
        valid_mask = torch.ones_like(topk_indices, dtype=torch.bool)  # (H, W, k_pad)

    # 创建一个映射：(y, x) -> index in mask_positions
    # 使用 coords_to_idx: (H, W)，不在 mask 中的设为 -1
    coords_to_idx = -torch.ones((H, W), dtype=torch.long, device=device)
    coords_to_idx[mask_positions[:, 0], mask_positions[:, 1]] = torch.arange(N, device=device)

    # 当前像素在 mask 中的 index（若不在则为 -1）
    self_indices = coords_to_idx[y_coords, x_coords]  # (H, W)

    # 扩展为 (H, W, k_pad)，判断 topk_indices 是否等于 self_indices
    self_idx_exp = self_indices.unsqueeze(-1).expand_as(topk_indices)  # (H, W, k_pad)
    is_self = (topk_indices == self_idx_exp)  # (H, W, k_pad), bool

    # 只有当前像素在 mask 中时，才需要剔除
    should_del = self_in_mask.unsqueeze(-1).expand_as(is_self)  # (H, W, k_pad)
    is_self_and_valid = is_self & should_del & valid_mask  # 仅当是自身且有效时才屏蔽

    # === 计算权重（指数衰减）===
    # d_max = torch.ones((H, W,k_pad), device=logits.device, dtype=torch.float32) * (H**2 + W**2) ** 0.5 * 0.5    # 半对角线长度
    d_max = topk_dists.max(dim=-1, keepdim=True).values
    # topk_dists_ = topk_dists.clone()
    # topk_dists_[topk_dists_ == 0] = float('inf')
    d_min = torch.zeros((H, W,k_pad), device=logits.device, dtype=torch.float32)
    eps = 1e-8
    sigma = (d_max - d_min + eps)
    weights = torch.exp(-(topk_dists - d_min) / sigma)  # (H, W, k_pad)
    # print(weights[5,5])

    # 屏蔽自身：如果当前像素在 mask 中且被选入 topk，则权重置0
    weights_masked = weights * (~is_self_and_valid).float()  # 剔除自身
    float_mask = valid_mask & ~is_self_and_valid
    weights_masked = torch.where(float_mask, weights_masked, torch.zeros_like(weights_masked))  # 同时保证 pad 和 self 都不参与
    # print(float_mask[5,5])

    total_weight = weights_masked.sum(dim=-1, keepdim=True)  # (H, W, 1)
    safe_weight = torch.where(total_weight > 0, total_weight, torch.ones_like(total_weight))
    # print(logits[11,8])
    # print(weights_masked[2,6])
    # print(topk_dists[2,6])
    # print(topk_logits[2,6])
    # print(safe_weight[5,5])

    # --- 情况1：不包含当前像素（已剔除）---
    weighted_sum_wo = (weights_masked * topk_logits).sum(dim=-1)
    mean_wo = weighted_sum_wo / safe_weight.squeeze(-1)
    # print(topk_logits[11,8])
    # print(mean_wo[11,8])
    # print(weighted_sum_wo[5,5])
    # print(mean_wo[5,5])

    mean_exp = mean_wo.unsqueeze(-1)
    var_numerator_wo = (weights_masked * (topk_logits - mean_exp) ** 2).sum(dim=-1)
    var_wo = torch.where(total_weight.squeeze(-1) > 0, var_numerator_wo / safe_weight.squeeze(-1), 0.0)
    # print(var_wo[11,8])

    # --- 情况2：包含当前像素 ---
    current_logit = logits.unsqueeze(-1)  # (H, W, 1)
    extended_logits = torch.cat([topk_logits, current_logit], dim=-1)  # (H, W, k_pad+1)

    # # 当前像素距离为0，计算其权重
    # current_d = torch.zeros_like(d_min)
    # current_weight_val = torch.exp(-current_d / sigma)  # (H, W, 1)

    # 注意：即使当前像素在 mask 中，我们也**重新加入它**（因为这是“包含”情况）
    extended_weights = torch.cat([weights_masked, total_weight], dim=-1)  # 用原始 weights（未剔除），再加 current

    # # 但 extended_weights 中的原始部分仍可能包含自身 → 我们希望在“包含”中它是独立加入的
    # # 所以：我们应确保 extended_weights 中原始部分不包含当前像素（避免重复）
    # # 方法：将原始 weights 中指向自身的也置0
    # weights_for_ext = weights * (~is_self_and_valid).float() * valid_mask.float()
    # # extended_weights_clean = torch.cat([weights_for_ext, current_weight_val], dim=-1)
    # extended_weights_clean = torch.cat([weights_for_ext, current_weight_val], dim=-1)

    ext_total_weight = extended_weights.sum(dim=-1, keepdim=True)
    safe_ext_weight = torch.where(ext_total_weight > 0, ext_total_weight, torch.ones_like(ext_total_weight))

    mean_w = (extended_weights * extended_logits).sum(dim=-1) / safe_ext_weight.squeeze(-1)
    mean_w_exp = mean_w.unsqueeze(-1)
    # print(mean_w_exp[11,8])
    var_numerator_w = (extended_weights * (extended_logits - mean_w_exp) ** 2).sum(dim=-1)
    var_w = torch.where(ext_total_weight.squeeze(-1) > 0, var_numerator_w / safe_ext_weight.squeeze(-1), 0.0)
    # print(var_w[11,8])

    return mean_wo, var_wo, mean_w, var_w

def compute_weighted_mean_variance_fast_v1(
    logits: torch.Tensor,
    mask_coords: torch.Tensor,        # [N, 2], precomputed
    mask_logits_vals: torch.Tensor,   # [N], precomputed
    coords_grid: torch.Tensor,        # [H*W, 2], precomputed once
    top_k: int = None,
    device: torch.device = None
):
    """
    Fast version for small images (<=64x64).
    Inputs are precomputed sparse representations.
    """
    if mask_coords.numel() == 0:
        H, W = logits.shape
        zero = torch.zeros_like(logits)
        return zero, zero, zero, zero

    H, W = logits.shape
    N = mask_coords.shape[0]
    HW = H * W

    # Flatten logits for easy indexing
    logits_flat = logits.view(-1)  # [HW]

    # All pixel coords: [HW, 2]
    all_pixel_coords = coords_grid  # precomputed

    # Compute distances: [HW, N]
    # For 64x64: HW=4096, N<=4096 → 16M elements, acceptable on GPU
    dists = torch.cdist(all_pixel_coords, mask_coords.float())  # [HW, N]

    k_use = min(top_k, N) if top_k is not None else N
    if k_use == 0:
        zero = torch.zeros_like(logits)
        return zero, zero, zero, zero

    # Top-k nearest (smallest distance)
    topk_dists, topk_indices = torch.topk(dists, k_use, dim=1, largest=False)  # [HW, k_use]

    # Gather logits values for top-k neighbors
    topk_logits = mask_logits_vals[topk_indices]  # [HW, k_use]

    # Determine if current pixel is in mask
    # Build a lookup: pixel_idx -> is_in_mask and its index in mask_coords
    pixel_idx_to_mask_idx = -torch.ones(HW, dtype=torch.long, device=device)
    flat_mask_idx = mask_coords[:, 0] * W + mask_coords[:, 1]  # [N]
    pixel_idx_to_mask_idx[flat_mask_idx] = torch.arange(N, device=device)

    current_pixel_idx = torch.arange(HW, device=device)
    self_in_mask = pixel_idx_to_mask_idx[current_pixel_idx] != -1  # [HW]
    self_mask_idx = pixel_idx_to_mask_idx[current_pixel_idx]       # [HW], -1 if not in mask

    # Expand for comparison
    self_mask_idx_exp = self_mask_idx.unsqueeze(1).expand(-1, k_use)  # [HW, k_use]
    topk_mask_indices = torch.arange(N, device=device)[topk_indices]   # [HW, k_use]
    is_self = (topk_mask_indices == self_mask_idx_exp) & self_in_mask.unsqueeze(1)

    # Compute weights (exponential decay)
    d_max = topk_dists.max(dim=1, keepdim=True).values  # [HW, 1]
    eps = 1e-8
    sigma = d_max + eps
    weights = torch.exp(-topk_dists / sigma)  # [HW, k_use]

    # Mask out self for "without self" case
    weights_wo = weights * (~is_self).float()
    total_weight_wo = weights_wo.sum(dim=1, keepdim=True)  # [HW, 1]
    safe_weight_wo = torch.where(total_weight_wo > 0, total_weight_wo, torch.ones_like(total_weight_wo))

    # Mean without self
    weighted_sum_wo = (weights_wo * topk_logits).sum(dim=1)  # [HW]
    mean_wo = weighted_sum_wo / safe_weight_wo.squeeze(1)

    # Variance without self
    var_numerator_wo = (weights_wo * (topk_logits - mean_wo.unsqueeze(1)) ** 2).sum(dim=1)
    var_wo = torch.where(total_weight_wo.squeeze(1) > 0, var_numerator_wo / safe_weight_wo.squeeze(1), 0.0)

    # For "with self": add current pixel as a new neighbor
    current_logit = logits_flat.unsqueeze(1)  # [HW, 1]
    current_weight = total_weight_wo  # heuristic: use total weight as current weight
    extended_logits = torch.cat([topk_logits, current_logit], dim=1)  # [HW, k_use+1]
    extended_weights = torch.cat([weights_wo, current_weight], dim=1)  # [HW, k_use+1]

    total_weight_w = extended_weights.sum(dim=1, keepdim=True)
    safe_weight_w = torch.where(total_weight_w > 0, total_weight_w, torch.ones_like(total_weight_w))
    mean_w = (extended_weights * extended_logits).sum(dim=1) / safe_weight_w.squeeze(1)
    var_numerator_w = (extended_weights * (extended_logits - mean_w.unsqueeze(1)) ** 2).sum(dim=1)
    var_w = torch.where(total_weight_w.squeeze(1) > 0, var_numerator_w / safe_weight_w.squeeze(1), 0.0)

    # Reshape to [H, W]
    mean_wo = mean_wo.view(H, W)
    var_wo = var_wo.view(H, W)
    mean_w = mean_w.view(H, W)
    var_w = var_w.view(H, W)

    return mean_wo, var_wo, mean_w, var_w

def compute_weighted_mean_variance_fast(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dist_matrix: torch.Tensor,
    coeff: float = 3.0
):
    """
    Compute distance-weighted mean and variance for each pixel using ALL mask points.
    
    Args:
        logits: (H, W)
        mask: (H, W), binary
        dist_matrix: (H*W, H*W), precomputed pairwise distances (flattened coordinates)
    
    Returns:
        mean_wo, var_wo, mean_w, var_w: each (H, W)
    """
    assert logits.shape == mask.shape
    H, W = logits.shape
    HW = H * W
    device = logits.device

    logits_flat = logits.view(HW)               # (HW,)
    mask_flat = mask.view(HW).bool()            # (HW,)
    # dist_matrix = dist_matrix.to(device)  # (HW, HW)

    mask_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(-1)  # (N,)
    N = mask_indices.numel()

    if N == 0:
        zero = torch.zeros_like(logits)
        return zero, zero, zero, zero

    # Distances from all pixels to mask points
    dists_to_mask = dist_matrix[:, mask_indices]  # (HW, N)
    mask_logits_vals = logits_flat[mask_indices]  # (N,)

    # Compute weights: exp(-d / (max_d + eps))
    d_max = dists_to_mask.max(dim=1, keepdim=True).values  # (HW, 1)
    eps = 1e-8
    weights_full = torch.exp(- coeff * dists_to_mask / (d_max + eps))  # (HW, N)

    # Identify if current pixel is in mask and appears in mask_indices
    self_in_mask = mask_flat  # (HW,)
    is_self = (mask_indices.unsqueeze(0) == torch.arange(HW, device=device).unsqueeze(1))  # (HW, N)
    is_self_and_valid = is_self & self_in_mask.unsqueeze(1)  # (HW, N)

    # --- Without self ---
    weights_wo = weights_full * (~is_self_and_valid).float()  # (HW, N)

    weight_sum_wo = weights_wo.sum(dim=1)  # (HW,)
    has_valid_wo = weight_sum_wo > 0

    # print(weights_wo[0]/weight_sum_wo[0])
    # # print(weighted_sum_wo[0])
    # a = input()

    # Compute mean_wo safely
    weighted_sum_wo = (weights_wo * mask_logits_vals.unsqueeze(0)).sum(dim=1)  # (HW,)
    mean_wo = torch.zeros_like(weighted_sum_wo)
    mean_wo[has_valid_wo] = weighted_sum_wo[has_valid_wo] / weight_sum_wo[has_valid_wo]

    # Compute var_wo
    var_wo = torch.zeros_like(weighted_sum_wo)
    if has_valid_wo.any():
        diff = mask_logits_vals.unsqueeze(0) - mean_wo.unsqueeze(1)  # (HW, N)
        var_num = (weights_wo * (diff ** 2)).sum(dim=1)  # (HW,)
        var_wo[has_valid_wo] = var_num[has_valid_wo] / weight_sum_wo[has_valid_wo]

    # --- With self: add current pixel with weight = weight_sum_wo (or 1 if no neighbors) ---
    current_logit = logits_flat  # (HW,)

    # 自身权重设计：若周围无点，则 w_self = 1；否则 w_self = weight_sum_wo
    # 这样保证至少有一个有效权重
    weight_self = torch.where(has_valid_wo, weight_sum_wo, torch.ones_like(weight_sum_wo))  # (HW,)

    # Total weight for "with self"
    weight_sum_w = weight_sum_wo + weight_self  # (HW,)

    # Mean_w = (W_wo * mean_wo + w_self * x_self) / (W_wo + w_self)
    mean_w = (weight_sum_wo * mean_wo + weight_self * current_logit) / weight_sum_w

    # Variance_w:
    # Var = [W_wo*(var_wo + (mean_wo - mean_w)^2) + w_self*(0 + (x_self - mean_w)^2)] / (W_wo + w_self)
    term1 = weight_sum_wo * (var_wo + (mean_wo - mean_w) ** 2)
    term2 = weight_self * (current_logit - mean_w) ** 2
    var_w = (term1 + term2) / weight_sum_w

    # Reshape
    return mean_wo.view(H, W), var_wo.view(H, W), mean_w.view(H, W), var_w.view(H, W)

def compute_weighted_mean_variance_fast_v3(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dist_matrix: torch.Tensor,
    top_k: int = 128,  # reasonable default
    coeff: float = 1.0
):
    """
    Efficient GPU implementation with top-k and precomputed distance matrix.
    Assumes logits is top-left (H, W) of 64x64 image.
    """
    assert logits.shape == mask.shape
    H, W = logits.shape
    HW = H * W
    device = logits.device

    logits_flat = logits.view(HW)
    mask_flat = mask.view(HW).bool()

    mask_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(-1)
    N = mask_indices.numel()

    if N == 0:
        zero = torch.zeros_like(logits)
        return zero, zero, zero, zero

    # Limit top_k to avoid OOM
    k_use = min(top_k, N)

    # --- Get top-k nearest mask points for each pixel ---
    # dists: (HW, N)
    dists_to_mask = dist_matrix[:, mask_indices]
    topk_dists, topk_idx_in_mask = torch.topk(dists_to_mask, k_use, dim=1, largest=False)  # (HW, k_use)

    # Gather logits of top-k points
    topk_logits = logits_flat[mask_indices[topk_idx_in_mask]]  # (HW, k_use)

    # --- Compute weights (exponential decay) ---
    d_max = topk_dists.max(dim=1, keepdim=True).values  # (HW, 1)
    eps = 1e-8
    weights = torch.exp(-coeff * topk_dists / (d_max + eps))  # (HW, k_use)

    # --- Identify if current pixel is in the top-k and in mask ---
    # Global index of current pixel: 0,1,...,HW-1
    current_global_idx = torch.arange(HW, device=device)  # (HW,)
    # Global index of top-k mask points:
    topk_global_idx = mask_indices[topk_idx_in_mask]  # (HW, k_use)
    # Is current pixel in mask?
    self_in_mask = mask_flat  # (HW,)
    # Is current pixel among the top-k neighbors?
    is_self_in_topk = (topk_global_idx == current_global_idx.unsqueeze(1))  # (HW, k_use)
    # Only mask out if it's both in mask AND in top-k
    should_mask_self = self_in_mask.unsqueeze(1) & is_self_in_topk  # (HW, k_use)

    # --- Case 1: Without self ---
    weights_wo = torch.where(should_mask_self, torch.zeros_like(weights), weights)
    weight_sum_wo = weights_wo.sum(dim=1)  # (HW,)
    has_valid_wo = weight_sum_wo > 0

    # Compute mean_wo
    weighted_sum_wo = (weights_wo * topk_logits).sum(dim=1)
    mean_wo = torch.zeros_like(weighted_sum_wo)
    mean_wo[has_valid_wo] = weighted_sum_wo[has_valid_wo] / weight_sum_wo[has_valid_wo]

    # Compute var_wo
    var_wo = torch.zeros_like(weighted_sum_wo)
    if has_valid_wo.any():
        diff = topk_logits - mean_wo.unsqueeze(1)
        var_num = (weights_wo * (diff ** 2)).sum(dim=1)
        var_wo[has_valid_wo] = var_num[has_valid_wo] / weight_sum_wo[has_valid_wo]

    # --- Case 2: With self ---
    current_logit = logits_flat
    # Self weight: use total weight of neighbors (or 1 if none)
    weight_self = torch.where(has_valid_wo, weight_sum_wo, torch.ones_like(weight_sum_wo))
    weight_sum_w = weight_sum_wo + weight_self

    mean_w = (weight_sum_wo * mean_wo + weight_self * current_logit) / weight_sum_w

    # Incremental variance
    term1 = weight_sum_wo * (var_wo + (mean_wo - mean_w) ** 2)
    term2 = weight_self * (current_logit - mean_w) ** 2
    var_w = (term1 + term2) / weight_sum_w

    return (
        mean_wo.view(H, W),
        var_wo.view(H, W),
        mean_w.view(H, W),
        var_w.view(H, W)
    )

def keep_negative_by_top2_magnitude_levels_old(x, target_size):
    """
    在 tensor 中，对负数部分：
    - 计算每个负数的十进制数量级（floor(log10(|x|))）
    - 找出最大的两个数量级
    - 保留属于这两个数量级的负数，其余负数置为 0
    - 非负数保持不变
    Args:
        x (torch.Tensor): 输入 tensor

    Returns:
        torch.Tensor: 处理后的 tensor
    """
    out = x.clone()

    # 找出负数
    negative_mask = x < 0
    if not negative_mask.any():
        return out  # 没有负数，直接返回

    negative_vals = x[negative_mask]

    # 计算绝对值
    abs_vals = negative_vals.abs()
    if negative_mask.float().sum() / target_size > 2:
        robust_max_idx = int(np.ceil(target_size * (1 - 0.8)))
        sorted_abs_vals = torch.sort(abs_vals.view(-1), descending=True).values
        robust_max = sorted_abs_vals[robust_max_idx]
    else:
        robust_max = torch.quantile(abs_vals, 0.90)
    abs_vals = torch.clamp_max(abs_vals, max=robust_max)
    # print(robust_max)

    # 安全处理：避免 log10(0)，但负数绝对值不会为0
    # 计算数量级：floor(log10(abs(x)))
    magnitudes = torch.floor(torch.log10(abs_vals)).to(torch.int)

    # 如果只有一个负数，直接保留
    if len(magnitudes) <= 2:
        return out

    # 找出唯二的数量级，排序（从大到小）
    unique_magnitudes = torch.unique(magnitudes, sorted=True)  # 升序
    top2_magnitudes = unique_magnitudes[-2:]  # 最大的两个数量级

    # 找出哪些负数属于这两个数量级
    keep_negative = (magnitudes.unsqueeze(1) == top2_magnitudes.unsqueeze(0)).any(dim=1)

    # 映射回原 tensor 的索引
    negative_indices = torch.nonzero(negative_mask, as_tuple=False).squeeze(1)
    keep_in_negative = negative_indices[keep_negative]

    # 构建最终保留 mask：非负数 + 属于 top2 数量级的负数
    final_keep_mask = torch.zeros_like(x, dtype=torch.bool)
    final_keep_mask[keep_in_negative[:, 0], keep_in_negative[:,1]] = True
    final_keep_mask |= (x > 0)  # 加上正数

    return out * final_keep_mask

def keep_negative_by_top2_magnitude_levels_old2(x, target_size):
    """
    在 tensor 中，对负数部分：
    - 计算每个负数的十进制数量级（floor(log10(|x|))）
    - 计算鲁棒最大值 robust_max（基于 target_size 的鲁棒截断或90%分位）
    - 取 robust_max 对应的数量级 level_max = floor(log10(robust_max))
    - 保留属于 level_max 和 level_max - 1 的负数，其余负数置为 0
    - 非负数保持不变

    Args:
        x (torch.Tensor): 输入 tensor
        target_size (int): 用于计算鲁棒最大值的参考尺寸（如总元素数）

    Returns:
        torch.Tensor: 处理后的 tensor
    """
    out = x.clone()

    # 找出负数
    negative_mask = x < 0
    if not negative_mask.any():
        return out  # 没有负数，直接返回

    negative_vals = x[negative_mask]
    abs_vals = negative_vals.abs()

    # --- 计算鲁棒最大值 ---
    if negative_mask.float().sum() / target_size > 2:
        robust_max_idx = torch.ceil(target_size * (1 - 0.8)).int()
        sorted_abs_vals = torch.sort(abs_vals.view(-1), descending=True).values
        robust_max = sorted_abs_vals[robust_max_idx] if robust_max_idx < len(sorted_abs_vals) else sorted_abs_vals[-1]
    else:
        robust_max = torch.quantile(abs_vals, 0.90)

    # 安全截断：避免异常大值影响数量级计算
    abs_vals = torch.clamp_max(abs_vals, max=robust_max)

    # --- 计算鲁棒最大值对应的数量级 ---
    # 注意：robust_max 可能为0？但负数绝对值>0，所以安全
    level_max = torch.floor(torch.log10(robust_max)).to(torch.int).item()

    # 定义要保留的数量级：level_max 和 level_max - 1
    keep_levels = {level_max, level_max - 1}

    # --- 计算每个负数的数量级 ---
    magnitudes = torch.floor(torch.log10(abs_vals)).to(torch.int)

    # --- 找出哪些负数属于 keep_levels ---
    keep_negative = torch.isin(magnitudes, torch.tensor(list(keep_levels), dtype=magnitudes.dtype, device=magnitudes.device))

    # --- 映射回原 tensor 的索引 ---
    negative_indices = torch.nonzero(negative_mask, as_tuple=False).squeeze(1)
    keep_in_negative = negative_indices[keep_negative]

    # --- 构建最终保留 mask ---
    final_keep_mask = torch.zeros_like(x, dtype=torch.bool)

    # 如果 keep_in_negative 是一维的（单通道），需要处理维度
    if keep_in_negative.dim() == 1:
        final_keep_mask[keep_in_negative] = True
    else:
        final_keep_mask[keep_in_negative[:, 0], keep_in_negative[:, 1]] = True

    # 加上非负数（>0 的部分）
    final_keep_mask |= (x >= 0)  # 注意：这里应该是 >=0，因为0也保留（原逻辑是>0，但0不是负数，也应该保留）

    return out * final_keep_mask

def keep_negative_by_top2_magnitude_levels(x, target_size):
    out = x.clone()
    negative_mask = x < 0
    if not negative_mask.any():
        return out

    negative_vals = x[negative_mask]
    abs_vals = negative_vals.abs()

    # --- 鲁棒最大值 ---
    neg_count = negative_mask.sum().float()
    if neg_count / target_size > 2:
        robust_max_idx = torch.ceil(target_size * 0.2).long()
        sorted_abs = torch.sort(abs_vals, descending=True).values
        robust_max = sorted_abs[min(robust_max_idx, len(sorted_abs) - 1)]
    else:
        robust_max = torch.quantile(abs_vals, 0.9)

    abs_vals = torch.clamp_max(abs_vals, robust_max)

    # --- 计算数量级（保持为张量）---
    log10_abs = torch.log10(abs_vals + 1e-12)  # 防止 log(0)
    magnitudes = torch.floor(log10_abs).long()  # [M]

    # --- 计算 level_max（张量）---
    level_max = torch.floor(torch.log10(robust_max + 1e-12)).long()

    # --- 构建保留掩码（向量化）---
    keep_mask_mag = (magnitudes == level_max) | (magnitudes == level_max - 1)

    # --- 映射回原图 ---
    final_keep = torch.zeros_like(x, dtype=torch.bool)
    neg_indices = torch.nonzero(negative_mask, as_tuple=False)  # [M, 2]
    if keep_mask_mag.any():
        keep_indices = neg_indices[keep_mask_mag]
        final_keep[keep_indices[:, 0], keep_indices[:, 1]] = True

    final_keep |= (x >= 0)
    return out * final_keep

def smooth_optim(logits, pos_bias=0.5):
    H, W = logits.shape
    # 先对logits进行映射
    credit_pos = (logits < 0.) * (-logits)
    credit_neg = (logits > 0.) * logits
    credit_pos_ = -(credit_pos / (credit_pos.max() + 1e-8) * (1 - pos_bias) + pos_bias * (logits < 0.))
    credit_neg_ = credit_neg / (credit_neg.max()+ 1e-8)
    credit = credit_pos_ + credit_neg_  #(H, W)

    padded_credit = F.pad(credit.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
    smooth_credit = gaussian_blurring_2D(padded_credit, 3, sigma=1)[1:1+H, 1:1+W]   

    smooth_credit_pos = smooth_credit * (smooth_credit < 0.0)
    smooth_credit_neg = smooth_credit * (smooth_credit > 0.0)
    smooth_credit_pos_ = (smooth_credit_pos - pos_bias * (logits < 0.)) / (1 - pos_bias) * credit_pos.max()
    smooth_credit_neg_ = smooth_credit_neg * credit_neg.max()
    new_logits = smooth_credit_pos_ + smooth_credit_neg_

    fig = plt.figure(figsize=(25, 5))
    plt.subplot(1, 5, 1)
    plt.imshow(logits, cmap='gray')
    plt.subplot(1, 5, 2)
    plt.imshow(credit, cmap='gray')
    plt.subplot(1, 5, 3)
    plt.imshow(smooth_credit, cmap='gray')
    plt.show(block=False)
    plt.subplot(1, 5, 4)
    plt.imshow(smooth_credit_pos_, cmap='gray')
    plt.subplot(1, 5, 5)
    plt.imshow(new_logits, cmap='gray')
    a = input()

    return new_logits

def add_uniform_points_cuda(mask, seed, num1, part_ratio=1, logits=None, chunk_size=2048):
    """
    CUDA-friendly version of adding uniformly distributed points.
    Uses chunked distance computation to avoid OOM and maximize GPU utilization.

    Args:
        mask (torch.Tensor): [H, W], bool, valid region
        seed (torch.Tensor): [H, W], bool, existing points
        num1 (int): number of new points to add
        part_ratio (float): ratio of points to add to the part of the image
        logits (torch.Tensor): [H, W], float, logits to select points part
        chunk_size (int): chunk size for distance computation to control memory

    Returns:
        new_seed (torch.Tensor): updated seed map with added points
    """
    assert mask.shape == seed.shape
    assert mask.dim() == 2
    device = mask.device
    dtype = torch.float32

    H, W = mask.shape

    # Convert to bool
    mask = mask.bool()
    seed = seed.bool()

    # Existing points: (N, 2) format (y, x)
    existing_points = torch.nonzero(seed, as_tuple=False).float()  # [N, 2]
    N = existing_points.shape[0]

    # Candidate points: in mask but not in seed
    candidate_mask = mask & (~seed)
    candidates = torch.nonzero(candidate_mask, as_tuple=False).float()  # [M, 2]
    M = candidates.shape[0]

    if num1 <= 0:
        return seed.clone()
    if M == 0:
        return seed.clone()
    if M < num1:
        num1 = M

    # Move to GPU if not already
    candidates = candidates.to(device, non_blocking=True)
    existing_points = existing_points.to(device, non_blocking=True)

    new_seed = torch.zeros_like(seed)
    added = 0

    # Pre-allocate for added points (for FPS-style update)
    all_selected = existing_points  # Will grow, but we avoid cat in loop if possible

    for _ in range(num1):
        if candidates.shape[0] == 0:
            break

        # Compute min distance from each candidate to all_selected
        min_dists = torch.full((candidates.shape[0],), float('inf'), device=device, dtype=dtype)

        # Chunked distance computation to avoid memory overflow
        for i in range(0, all_selected.shape[0], chunk_size):
            batch = all_selected[i:i+chunk_size]  # [B, 2]
            # Compute distance: candidates [M', 2] vs batch [B, 2]
            dists = torch.cdist(candidates, batch)  # [M', B]
            min_dists = torch.min(min_dists, torch.min(dists, dim=1).values)  # update min

        # Find candidate with maximum of min distances
        if min_dists.numel() == 0:
            break
        idx = torch.argmax(min_dists)

        # Selected point (in float format)
        selected_float = candidates[idx]  # [2], (y, x)
        selected_int = selected_float.round().long()
        y, x = selected_int[0].item(), selected_int[1].item()

        # Update new_seed
        new_seed[y, x] = True
        added += 1

        # Add to all_selected
        all_selected = torch.cat([all_selected, selected_float.unsqueeze(0)], dim=0)

        # Remove selected from candidates
        mask_out = torch.any(candidates != selected_float, dim=1)
        candidates = candidates[mask_out]

    updated_seed = seed.clone()

    if part_ratio < 1:
        candidates_seed_logits = logits * new_seed
        k = int(num1 * part_ratio)
        k = k if k > 0 else 1
        _, idx = torch.topk(candidates_seed_logits.view(-1), k)
        for j in idx:
            updated_seed[j // W, j % W] = True
    else:
        updated_seed = updated_seed | new_seed

    return updated_seed

def add_uniform_points_grid_cuda_v1(mask, seed, num, jitter=True):
    """
    Add 'num' uniformly distributed points inside mask, avoiding seed,
    using jittered grid + farthest selection (GPU optimized).

    Args:
        mask (torch.Tensor): [H, W], bool, valid region
        seed (torch.Tensor): [H, W], bool, existing points
        num (int): number of new points to add
        jitter (bool): whether to add random jitter in each grid cell

    Returns:
        torch.Tensor: updated seed map, same shape as input
    """
    device = mask.device
    H, W = mask.shape

    # Ensure boolean
    mask = mask.bool()
    seed = seed.bool()

    # Existing seed points (float format for distance computation)
    existing_points = torch.nonzero(seed, as_tuple=False).float()  # [N, 2] (y, x)
    N = existing_points.shape[0]

    # Candidate area: mask but not seed
    candidate_mask = mask & (~seed)
    if num <= 0:
        return seed.clone()
    if not candidate_mask.any():
        return seed.clone()

    # -------------------------------
    # Step 1: Generate jittered grid candidates
    # -------------------------------
    n_side = int(num ** 0.5) + 2  # Slightly over-sample
    gy = torch.linspace(0, H - 1e-5, n_side + 1, device=device)  # Avoid edge issues
    gx = torch.linspace(0, W - 1e-5, n_side + 1, device=device)

    # 替换整个 grid 生成部分
    gy = torch.linspace(0, H - 1e-5, n_side + 1, device=device)
    gx = torch.linspace(0, W - 1e-5, n_side + 1, device=device)

    # 生成网格坐标 (n_side, n_side)
    gy_mid = (gy[:-1] + gy[1:]) / 2  # [n_side]
    gx_mid = (gx[:-1] + gx[1:]) / 2  # [n_side]

    # meshgrid
    Y, X = torch.meshgrid(gy_mid, gx_mid, indexing='ij')  # [n_side, n_side]
    candidates_y = Y.flatten()  # [K]
    candidates_x = X.flatten()  # [K]

    if jitter:
        dy = torch.rand_like(candidates_y) * (gy[1] - gy[0])
        dx = torch.rand_like(candidates_x) * (gx[1] - gx[0])
        candidates_y += dy
        candidates_x += dx

    candidates = torch.stack([candidates_y, candidates_x], dim=1)  # [K, 2]
    K = candidates.shape[0]

    # Convert to integer coordinates for indexing
    cand_int = candidates.long()  # [K, 2] (y, x)

    # Filter: within bounds
    valid = (cand_int[:, 0] < H) & (cand_int[:, 1] < W) & (cand_int[:, 0] >= 0) & (cand_int[:, 1] >= 0)
    candidates = candidates[valid]
    cand_int = cand_int[valid]

    # Further filter: must be in mask and not in seed
    spatial_valid = mask[cand_int[:, 0], cand_int[:, 1]] & (~seed[cand_int[:, 0], cand_int[:, 1]])
    candidates = candidates[spatial_valid]
    cand_int = cand_int[spatial_valid]

    # If not enough grid points, supplement with random samples from candidate_mask
    current_num = candidates.shape[0]
    if current_num < num:
        extra_coords = torch.nonzero(candidate_mask, as_tuple=False).float()
        if extra_coords.shape[0] > 0:
            # Remove duplicates (based on int coords)
            all_existing_int = torch.cat([
                torch.nonzero(seed, as_tuple=False),
                cand_int
            ], dim=0)
            # Use set logic via unique
            grid_or_seed = torch.zeros_like(candidate_mask)
            for y, x in torch.unique(all_existing_int, dim=0):
                if 0 <= y < H and 0 <= x < W:
                    grid_or_seed[y, x] = True
            extra_candidate_mask = candidate_mask & (~grid_or_seed)
            extra_coords = torch.nonzero(extra_candidate_mask, as_tuple=False).float()

            if extra_coords.shape[0] > 0:
                # Randomly sample up to (num - current_num)
                n_extra = min(num - current_num, extra_coords.shape[0])
                perm = torch.randperm(extra_coords.shape[0], device=device)[:n_extra]
                extra_candidates = extra_coords[perm]
                candidates = torch.cat([candidates, extra_candidates], dim=0)
                cand_int = torch.cat([cand_int, extra_candidates.long()], dim=0)

    # Now we have at least `num` candidates (or fewer if very constrained)
    if candidates.shape[0] == 0:
        return seed.clone()

    # -------------------------------
    # Step 2: Select top-`num` points that are farthest from existing seed
    # -------------------------------
    if N > 0 and candidates.shape[0] > 0:
        # Compute min distance from each candidate to any existing point
        dists = torch.cdist(candidates, existing_points)  # [K', N]
        min_dists = dists.min(dim=1).values  # [K']
        # Select the `num` points with largest min-distance
        k_select = min(num, min_dists.shape[0])
        _, idx_selected = torch.topk(min_dists, k=k_select, largest=True)
    else:
        # No existing points or no candidates -> random selection
        k_select = min(num, candidates.shape[0])
        idx_selected = torch.randperm(candidates.shape[0], device=device)[:k_select]

    final_cand_int = cand_int[idx_selected]  # [k_select, 2]

    # -------------------------------
    # Step 3: Update seed map
    # -------------------------------
    new_seed = seed.clone()
    # Use advanced indexing for batch write (efficient on GPU)
    ys = final_cand_int[:, 0]
    xs = final_cand_int[:, 1]
    valid_update = (ys < H) & (xs < W) & (ys >= 0) & (xs >= 0)
    ys = ys[valid_update]
    xs = xs[valid_update]
    new_seed[ys, xs] = True

    return new_seed

def add_uniform_points_grid_cuda(mask, seed, num, min_dist_ratio=0.5):
    """
    add_uniform_points_nms
    Add 'num' approximately uniformly distributed points in mask (excluding seed)
    using random sampling + distance-based Non-Maximum Suppression (NMS).
    
    This approximates Poisson disk sampling by ensuring selected points are not too close.
    
    Args:
        mask (torch.Tensor): [H, W], bool, valid region
        seed (torch.Tensor): [H, W], bool, existing points
        num (int): number of new points to add
        min_dist_ratio (float): controls min distance as ratio of avg grid spacing.
                                Smaller -> denser; larger -> sparser.
                                Typical: 0.3 ~ 0.7.

    Returns:
        torch.Tensor: updated seed map [H, W], bool
    """
    if num <= 0:
        return seed.clone()
    
    mask = mask.bool()
    seed = seed.bool()
    H, W = mask.shape
    device = mask.device

    candidate_mask = mask & (~seed)
    candidate_coords = torch.nonzero(candidate_mask, as_tuple=False).float()  # [M, 2]
    M = candidate_coords.shape[0]

    if M == 0:
        return seed.clone()
    
    # If not enough candidates, take all
    if M <= num:
        selected = candidate_coords
    else:
        # Step 1: Oversample candidates (e.g., 3x~5x)
        oversample = min(5 * num, M)
        idx = torch.randperm(M, device=device)[:oversample]
        candidates = candidate_coords[idx]  # [K, 2], K = oversample
        
        # Step 2: Estimate a reasonable min distance
        # Approximate average spacing if points were uniform over mask area
        mask_area = mask.sum().float()
        if mask_area > 0:
            avg_spacing = (mask_area ** 0.5) / (num ** 0.5 + 1e-6)
        else:
            avg_spacing = min(H, W) * 0.1
        min_dist = min_dist_ratio * avg_spacing

        # Step 3: Distance-based NMS
        # Start with all candidates as potential
        keep = torch.ones(candidates.shape[0], dtype=torch.bool, device=device)
        selected = []

        # Compute pairwise distances once (K can be ~500, so K^2 is acceptable)
        dists = torch.cdist(candidates, candidates)  # [K, K]

        # Greedily select: pick highest-score (here, all equal) and suppress neighbors
        # Since no score, we just iterate in order (or shuffle for randomness)
        order = torch.randperm(candidates.shape[0], device=device)
        for i in order:
            if not keep[i]:
                continue
            selected.append(candidates[i:i+1])
            if len(selected) >= num:
                break
            # Suppress all points within min_dist
            too_close = dists[i] < min_dist
            keep = keep & (~too_close)

        if selected:
            selected = torch.cat(selected, dim=0)
        else:
            # Fallback: just take first 'num'
            selected = candidates[:num]

    # Update seed map
    new_seed = seed.clone()
    ys = selected[:, 0].long()
    xs = selected[:, 1].long()
    valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    new_seed[ys[valid], xs[valid]] = True
    return new_seed

def add_uniform_points_with_logits(mask, seed, logits, num, alpha=0.6, jitter=True):
    """
    Add 'num' points in mask region using adaptive grid based on mask area.
    Generates ~2*num candidates to handle irregular shapes, then selects top-num by score.

    Args:
        mask (torch.Tensor): [H, W], bool
        seed (torch.Tensor): [H, W], bool
        logits (torch.Tensor): [H, W], float
        num (int): number of points to add
        alpha (float): weight for spatial uniformity (0~1)
        jitter (bool): whether to add random jitter within each cell

    Returns:
        torch.Tensor: updated seed map
    """
    device = mask.device
    H, W = mask.shape

    assert mask.shape == seed.shape == logits.shape
    assert 0 <= alpha <= 1
    if num <= 0:
        return seed.clone()

    # -------------------------------
    # Step 0: Prepare candidate area
    # -------------------------------
    mask = mask.bool()
    seed = seed.bool()
    candidate_mask = mask & (~seed)

    if not candidate_mask.any():
        return seed.clone()

    # -------------------------------
    # Step 1: Compute adaptive grid cell size from mask area
    # -------------------------------
    area = mask.sum().float().item()
    if area < 1e-3:
        num = torch.ceil(area).int()

    # Estimate ideal cell size: sqrt(area / num)
    cell_size = (area / num) ** 0.5
    cell_size = max(1.0, cell_size)  # At least 1x1

    # Number of cells in H and W direction
    nx = int(W / cell_size) + 2
    ny = int(H / cell_size) + 2

    # Grid boundaries
    x_edges = torch.linspace(0, W, nx + 1, device=device)
    y_edges = torch.linspace(0, H, ny + 1, device=device)

    candidates = []

    # 替换原来的 for 循环
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2  # [ny]
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2  # [nx]

    if jitter:
        y_jitter = (torch.rand_like(y_centers) * cell_size)
        x_jitter = (torch.rand_like(x_centers) * cell_size)
        y_centers = y_centers + y_jitter
        x_centers = x_centers + x_jitter

    # 扩展为网格点
    yy, xx = torch.meshgrid(y_centers, x_centers, indexing='ij')  # [ny, nx]
    candidates = torch.stack([yy.flatten(), xx.flatten()], dim=1)  # [K, 2]

    # Filter valid indices
    cand_int = candidates.long()  # [K, 2] (y, x)
    valid_pos = (cand_int[:, 0] < H) & (cand_int[:, 1] < W) & (cand_int[:, 0] >= 0) & (cand_int[:, 1] >= 0)
    candidates = candidates[valid_pos]
    cand_int = cand_int[valid_pos]

    # Further filter: must be in candidate_mask
    ys, xs = cand_int[:, 0], cand_int[:, 1]
    in_candidate = candidate_mask[ys, xs]
    candidates = candidates[in_candidate]
    cand_int = cand_int[in_candidate]
    logits_vals = logits[ys, xs][in_candidate]

    if candidates.shape[0] == 0:
        return seed.clone()

    # -------------------------------
    # Step 2: Oversample strategy — keep up to 2*num candidates
    # -------------------------------
    K = candidates.shape[0]
    target_keep = min(K, 2 * num)  # Aim for ~2*num

    # Optional: sort by spatial spread or logits before truncating?
    # We'll keep them as-is (random order due to grid), then re-rank by score later

    if K > target_keep:
        # Randomly subsample to avoid OOM in distance computation
        perm = torch.randperm(K, device=device)[:target_keep]
        candidates = candidates[perm]
        cand_int = cand_int[perm]
        logits_vals = logits_vals[perm]

    # -------------------------------
    # Step 3: Compute combined score
    # -------------------------------
    scores = torch.zeros(candidates.shape[0], device=device, dtype=torch.float32)

    # Term 1: Distance to nearest existing seed (if any)
    has_existing = seed.any()
    if has_existing and alpha > 0.0:
        existing_points = torch.nonzero(seed, as_tuple=False).float()
        dists = torch.cdist(candidates, existing_points)
        min_dists = dists.min(dim=1).values
        max_dist = min_dists.max() + 1e-8
        norm_dist = min_dists / max_dist
        scores += alpha * norm_dist

    # Term 2: Logits importance
    logit_score = torch.sigmoid(logits_vals)  # Normalize to [0,1]
    scores += (1 - alpha) * logit_score

    # -------------------------------
    # Step 4: Select top-`num` points
    # -------------------------------
    k_select = min(num, scores.shape[0])
    _, idx_selected = torch.topk(scores, k=k_select, largest=True)
    final_cand_int = cand_int[idx_selected]

    # -------------------------------
    # Step 5: Update seed map
    # -------------------------------
    new_seed = seed.clone()
    ys_new, xs_new = final_cand_int[:, 0], final_cand_int[:, 1]
    valid_update = (ys_new < H) & (xs_new < W) & (ys_new >= 0) & (xs_new >= 0)
    ys_new = ys_new[valid_update]
    xs_new = xs_new[valid_update]
    new_seed[ys_new, xs_new] = True

    return new_seed

def topk_mask(x, num, largest=True):
    """
    返回一个 mask，标记 Tensor 中最大的 num 个元素的位置。
    
    参数:
        x (torch.Tensor): 输入张量
        num (int): 要选取的最大元素个数
    
    返回:
        torch.Tensor: 布尔类型 mask，形状与 x 相同
    """
    if num <= 0:
        return torch.zeros_like(x, dtype=torch.bool)
    
    # 展平张量以进行排序
    flat_x = x.flatten()
    
    # 处理 num 大于元素总数的情况
    num = min(num, flat_x.numel())
    
    # 获取最大的 num 个值的索引
    _, indices = torch.topk(flat_x, num, largest=largest)
    
    # 创建与 x 形状相同的全 False mask
    mask = torch.zeros_like(flat_x, dtype=torch.bool)
    
    # 将 top-k 索引位置设为 True
    mask[indices] = True
    
    # 恢复 mask 到原始形状
    return mask.reshape(x.shape)

def big_num_mask(x, num, largest=True):
    """
    返回一个 mask，标记 Tensor 中最大的 num 个元素的位置。
    
    参数:
        x (torch.Tensor): 输入张量
        num (int): 要选取的最大元素个数
    
    返回:
        torch.Tensor: 布尔类型 mask，形状与 x 相同
    """
    if num <= 0:
        return torch.zeros_like(x, dtype=torch.bool)
    
    # 展平张量以进行排序
    flat_x = x.flatten()
    
    # 处理 num 大于元素总数的情况
    num = min(num, flat_x.numel())

    # num = int(num * 0.8)
    
    # 获取最大的 num 个值的索引
    vals, indices = torch.topk(flat_x, num, largest=largest)
    if largest:
        # vals = vals.min() / 2.5
        # vals = 0 if vals < 0 else vals
        result = x >= vals.min()
    else:
        # vals = vals.max() / 2.5
        # vals = 0 if vals > 0 else vals
        result = x <= vals.max()
    
    return result 

def erode_mask_4connectivity(mask, d=1):
    """
    对 mask 进行基于 4-邻域的腐蚀：
    仅当像素自身为1，且其上、下、左、右均为1时，才保留为1。
    使用卷积实现，结构元为十字形（+），大小由 d 控制。
    
    Args:
        mask (Tensor): [H, W], dtype=torch.float, 值为0或1
        d (int): 邻域半径，控制十字臂长。d=1 表示紧邻的上下左右
    
    Returns:
        eroded (Tensor): [H, W], 腐蚀后的 mask
    """
    H, W = mask.shape
    device = mask.device

    # 创建十字形结构元: 中心行和中心列全1
    kernel_size = 2 * d + 1
    weight = torch.zeros((1, 1, kernel_size, kernel_size), device=device)
    center = d
    weight[0, 0, :, center] = 1  # 垂直方向（上下）
    weight[0, 0, center, :] = 1  # 水平方向（左右）

    # 总共激活的元素个数（用于判断是否全为1）
    expected_sum = kernel_size * 2 - 1  # 行 + 列 - 重复的中心

    mask_float = mask.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # 卷积操作
    conv_result = F.conv2d(mask_float, weight, padding=d)

    # 只有当卷积结果等于 expected_sum 且中心像素也为1时，才保留
    center_mask = mask  # 中心必须为1
    valid = (conv_result.squeeze() == expected_sum).float()
    eroded = valid * center_mask

    return eroded

def add_uniform_points_v2(mask, seed, num1):
    credit_mask = erode_mask(mask, 1)
    target_mask = credit_mask * ~seed
    if target_mask.sum() > 0:
        return add_uniform_points_cuda(credit_mask, seed, num1)
    credit_mask = erode_mask_4connectivity(mask, 1)
    target_mask = credit_mask * ~seed
    if target_mask.sum() > 0:
        return add_uniform_points_cuda(credit_mask, seed, num1)
    return add_uniform_points_cuda(mask, seed, num1)

def add_uniform_points_v3(logits, mask, seed, num1, mode):
    add_uniform_points_cuda = add_uniform_points_grid_cuda
    if mode == "fg":
        credit_mask = mask * (logits > 0.8)
        target_mask = credit_mask * ~seed
        if target_mask.sum() > 0:
            return add_uniform_points_cuda(credit_mask, seed, num1)
        credit_mask = mask * (logits > 0.5)
        target_mask = credit_mask * ~seed
        if target_mask.sum() > 0:
            return add_uniform_points_cuda(credit_mask, seed, num1)
        credit_mask = mask * (logits > 0.1)
        target_mask = credit_mask * ~seed
        if target_mask.sum() > 0:
            return add_uniform_points_cuda(credit_mask, seed, num1)
        return add_uniform_points_cuda(mask, seed, num1)
    else:
        credit_mask = mask * (logits < 0.02)
        target_mask = credit_mask * ~seed
        if target_mask.sum() > 0:
            return add_uniform_points_cuda(credit_mask, seed, num1)
        credit_mask = mask * (logits < 0.05)
        target_mask = credit_mask * ~seed
        if target_mask.sum() > 0:
            return add_uniform_points_cuda(credit_mask, seed, num1)
        credit_mask = mask * (logits < 0.1)
        target_mask = credit_mask * ~seed
        if target_mask.sum() > 0:
            return add_uniform_points_cuda(credit_mask, seed, num1)
        return add_uniform_points_cuda(mask, seed, num1)

def get_min_value_outermost_mask(tensor):
    """
    输入: tensor, shape [H, W]
    输出: mask, shape [H, W], 类型 torch.bool 或 torch.float
          只有一个位置为 1，对应最小值中最靠近边界的那个点。
    """
    H, W = tensor.shape
    
    # 步骤1: 找到最小值
    min_value = tensor.min()
    
    # 步骤2: 找到所有等于最小值的位置
    min_positions = (tensor == min_value)  # bool mask
    coords = torch.nonzero(min_positions)  # [num_min, 2], 每行是 (i, j)
    
    if coords.numel() == 0:
        raise ValueError("No minimum value found")

    # 步骤3: 计算每个最小值位置到图像边界的距离
    i_coords = coords[:, 0]  # shape: [num_min]
    j_coords = coords[:, 1]  # shape: [num_min]
    
    # 到四个边的距离：上、下、左、右
    dist_to_boundary = torch.stack([
        i_coords,           # 到上边
        H - 1 - i_coords,   # 到下边
        j_coords,           # 到左边
        W - 1 - j_coords    # 到右边
    ], dim=1)  # shape: [num_min, 4]

    # 每个点的最小距离（越小越靠外）
    min_dist = dist_to_boundary.min(dim=1).values  # shape: [num_min]

    # 步骤4: 找到最小距离中的最小值（最外围），如果有多个，取第一个
    outermost_idx = min_dist.argmin()  # 第一个最外围的索引
    selected_coord = coords[outermost_idx]  # [2], (i, j)

    # 步骤5: 构造 mask
    mask = torch.zeros(H, W, dtype=torch.bool, device=tensor.device)
    mask[selected_coord[0], selected_coord[1]] = True

    return mask

def periodic_function(t, period, amplitude, phase=0.0):
    """
    计算周期函数值：f(t) = a * wave((2π / p) * t + φ)

    Args:
        t (float or torch.Tensor): 当前时间
        period (float): 周期 p
        amplitude (float): 振幅 a
        phase (float): 相位偏移（弧度），默认0

    Returns:
        float or torch.Tensor: 周期函数值
    """
    # 角频率
    omega = 2 * math.pi / period
    x = omega * t + phase  # 相位

    return amplitude * (torch.sin(x) + 1) / 2

def bilateral_smooth_logits(logits, image, sigma_spatial=5.0, sigma_value=0.1):
    """
    使用 image 作为引导图，对 logits 进行双边平滑。
    权重 = 空间高斯权重 * 值域高斯权重

    Args:
        logits: [H, W] 的 tensor
        image: [H, W] 的 tensor，作为引导图
        sigma_spatial: 空间距离的标准差（控制多远的像素参与加权）
        sigma_value: 像素值差异的标准差（控制多相似的像素值才参与加权）

    Returns:
        smoothed_logits: [H, W] 平滑后的 logits
    """
    H, W = logits.shape
    device = logits.device

    # 创建空间坐标网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )  # [H, W]

    # 展平所有坐标和像素值
    y_flat = y_coords.reshape(-1)  # [H*W]
    x_flat = x_coords.reshape(-1)  # [H*W]
    image_flat = image.reshape(-1)  # [H*W]
    logits_flat = logits.reshape(-1)  # [H*W]

    # 计算空间距离矩阵 (H*W, H*W)
    dy = y_flat.unsqueeze(1) - y_flat.unsqueeze(0)  # [H*W, H*W]
    dx = x_flat.unsqueeze(1) - x_flat.unsqueeze(0)
    spatial_dist_sq = dx**2 + dy**2  # [H*W, H*W]

    # 计算值域差异矩阵
    value_diff = image_flat.unsqueeze(1) - image_flat.unsqueeze(0)  # [H*W, H*W]
    value_diff_sq = value_diff ** 2

    # 计算双边权重
    spatial_weight = torch.exp(-spatial_dist_sq / (2 * sigma_spatial**2))
    value_weight = torch.exp(-value_diff_sq / (2 * sigma_value**2))
    bilateral_weight = spatial_weight * value_weight  # [H*W, H*W]

    # 归一化权重（每行和为1）
    weight_sum = bilateral_weight.sum(dim=1, keepdim=True)  # [H*W, 1]
    bilateral_weight = bilateral_weight / (weight_sum + 1e-8)  # 避免除零

    # 加权平均 logits
    smoothed_logits_flat = torch.matmul(bilateral_weight, logits_flat.unsqueeze(1)).squeeze(1)  # [H*W]

    # 恢复形状
    smoothed_logits = smoothed_logits_flat.view(H, W)

    return smoothed_logits

def check_cube(mask: torch.Tensor):
    """
    在二值 mask 中查找是否存在孤立的 3x3 全 1 矩形区域。
    “孤立”定义：该 3x3 区域周围一圈（外扩1像素）必须全为 0（或越界）。
    
    参数:
        mask (torch.Tensor): 形状为 [H, W] 的二值张量（值为 0 或 1）
    
    返回:
        torch.Tensor: 如果找到，返回该 3x3 区域；
                     如果未找到，返回 None
    """
    assert mask.ndim == 2, "mask must be 2D tensor [H, W]"
    H, W = mask.shape

    if H < 3 or W < 3:
        return None

    # Step 1: 找出所有 3x3 全1区域（卷积值=9）
    kernel_3x3 = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device)
    mask_f = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
    conv_3x3 = F.conv2d(mask_f, kernel_3x3, stride=1, padding=1)  # [1, 1, H, W]
    candidates = (conv_3x3.squeeze() == 9.0)  # [H, W]

    if not candidates.any():
        return None

    # Step 2: 构造 5x5 外圈检测核（只检测外圈，中心3x3为0）
    # 创建 5x5 核：外圈为1，内3x3为0
    kernel_5x5_outer = torch.ones((1, 1, 5, 5), dtype=torch.float32, device=mask.device)
    kernel_5x5_outer[0, 0, 1:4, 1:4] = 0  # 中心3x3置0

    # 对原图做 5x5 卷积（检测每个 5x5 区域外圈是否有1）
    conv_5x5_outer = F.conv2d(mask_f, kernel_5x5_outer, stride=1, padding=2)  # [1, 1, H, W]
    outer1_sum = conv_5x5_outer.squeeze()  # [H-4, W-4]

    # Step 3: 构造 7x7 外圈检测核（只检测外圈，中心5x5为0）
    # 创建 7x7 核：外圈为1，内5x5为0
    kernel_7x7_outer = torch.ones((1, 1, 7, 7), dtype=torch.float32, device=mask.device)
    kernel_7x7_outer[0, 0, 1:6, 1:6] = 0  # 中心5x5置0

    # 对原图做 7x7 卷积（检测每个 7x7 区域外圈是否有1）
    conv_7x7_outer = F.conv2d(mask_f, kernel_7x7_outer, stride=1, padding=3)  # [1, 1, H, W]
    outer2_sum = conv_7x7_outer.squeeze()  # [H-4, W-4]

    isolated = candidates & (outer1_sum <= 12) & (outer2_sum <= 1)

    return isolated

