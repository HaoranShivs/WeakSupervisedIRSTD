import torch
import torch.nn.functional as F

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from scipy.ndimage import label as ndlabel

from utils.utils import gaussian_kernel


def object_closed_score(region, width_scale): 
    H, W = region.shape
    min_size = min(H, W)
    width = min_size // width_scale
    width = width if width > 0 else 1
    top = region[:width]
    bottom = region[-width:]
    left = region[:, :width]
    right = region[:, -width:]

    def score_map(score):
        return 10 ** (-score*2)

    top_score = torch.sum(top) / (W * width)
    bottom_score = torch.sum(bottom) / (W * width)
    left_score = torch.sum(left) / (H * width)
    right_score = torch.sum(right) / (H * width)
    score = max(top_score, bottom_score, left_score, right_score)

    return score_map(score)

def object_closed_score_v2(region, width_scale): 
    H, W = region.shape
    H_width = H // width_scale
    W_width = W // width_scale
    H_width = H_width if H_width > 0 else 1
    W_width = W_width if W_width > 0 else 1
    middle_area = region[H_width:-H_width, W_width:-W_width]

    score = torch.sum(region) / torch.sum(middle_area) - 1

    return torch.exp(-score)

def score_local_region(region, ideal_ratio_low=0.125, ideal_ratio_high=0.25, width_scale=4.0):
    """
    评估局部区域 region 的“合适程度”，用于小目标伪标签裁剪。
    
    得分基于两个因素：
      1. 目标物体在 region 中的空间集中度（高斯加权中心性）
      2. 目标物体占 region 面积的比例是否在理想区间 [ideal_ratio_low, ideal_ratio_high]
    
    参数:
        region (torch.Tensor): [H, W] 二值张量，1 表示目标物体
        ideal_ratio_low (float): 理想占比下限（如 0.125）
        ideal_ratio_high (float): 理想占比上限（如 0.25）
        width_scale (float): 控制高斯中心敏感度，越大越宽松
    
    返回:
        score (torch.Tensor): 标量，范围 [0, 1]，越高表示越合适
    """
    H, W = region.shape
    device = region.device

    total_pixels = H * W
    object_pixels = region.sum().float()

    if object_pixels == 0:
        return torch.tensor(0.0, device=device)

    # ===== 1. 计算空间集中度得分（高斯加权） =====
    i_coords = torch.arange(H, device=device).float()
    j_coords = torch.arange(W, device=device).float()
    I, J = torch.meshgrid(i_coords, j_coords, indexing='ij')

    center_i = (H - 1) / 2.0
    center_j = (W - 1) / 2.0

    dist_i = torch.abs(I - center_i) / (H / 2.0)
    dist_j = torch.abs(J - center_j) / (W / 2.0)
    dist = torch.sqrt(dist_i**2 + dist_j**2)

    sigma = 1.0 / width_scale
    weight_map = torch.exp(-0.5 * (dist / sigma)**2)

    concentration_score = (region.float() * weight_map).sum() / object_pixels

    # ===== 2. 计算面积比例得分（高斯钟形曲线） =====
    current_ratio = object_pixels / total_pixels

    ideal_center = (ideal_ratio_low + ideal_ratio_high) / 2.0  # 0.1875
    ideal_half_width = (ideal_ratio_high - ideal_ratio_low) / 2.0  # 0.0625

    # 使用高斯函数：峰值在 ideal_center，标准差设为 ideal_half_width
    ratio_score = torch.exp(-0.5 * ((current_ratio - ideal_center) / ideal_half_width)**2)

    # ===== 3. 综合得分 =====
    final_score = concentration_score * ratio_score

    return final_score


def finalize_target(target_l, view=False):
    """
    最合适的尺度选择
    """
    scores = []
    for _target in target_l:
        target = _target['target']
        # y1, x1, y2, x2 = _target['coor']
        target_closed_score = object_closed_score_v2(target, 4)
        # target_closed_score = score_local_region(target, ideal_ratio_high=0.125, ideal_ratio_low=0.03125, width_scale=0.5)
        scores.append(target_closed_score)
        # output_[y1:y2, x1:x2] = output_[y1:y2, x1:x2] + target * target_closed_score
    # output_ = (output_ - output_.min())/(output_.max() - output_.min())
    # print(scores)
    idx = np.argmax(scores)
    return target_l[idx]['target'], scores, target_l[idx]['coor']

def compute_histogram(image, bins=256, range=(0, 256)):
    """
    计算图像的亮度直方图。

    参数:
        image (torch.Tensor): 输入图像张量，形状为 [H, W] 或 [C, H, W]，值范围在 [0, 255]。
        bins (int): 直方图的区间数，默认为 256。
        range (tuple): 像素值的范围，默认为 (0, 256)。

    返回:
        hist (torch.Tensor): 直方图，长度为 `bins`。
        bin_edges (torch.Tensor): 每个区间的边界，长度为 `bins + 1`。
    """
    # 确保输入是 [H, W] 形状
    if image.dim() == 3:  # 如果是 [C, H, W]，取第一个通道
        image = image[0]
    elif image.dim() != 2:
        raise ValueError("Input image must be of shape [H, W] or [C, H, W].")

    # 展平图像为一维张量
    flat_image = image.flatten()

    # 创建区间边界
    bin_edges = torch.linspace(range[0], range[1], bins)

    # 计算每个像素所属的区间索引
    indices = torch.bucketize(flat_image, bin_edges)

    # 统计每个区间的像素数量
    hist = torch.zeros(bins, dtype=torch.int32)
    hist.index_add_(dim=0, index=indices, source=torch.ones_like(indices, dtype=torch.int32))

    return hist, bin_edges

def smooth_histogram_gaussian(hist, sigma=1.0):
    """
    使用高斯滤波平滑直方图。

    参数:
        hist (torch.Tensor): 输入直方图，形状为 [bins]。
        sigma (float): 高斯核的标准差，默认为 1.0。

    返回:
        torch.Tensor: 平滑后的直方图。
    """
    # 定义高斯核大小（通常选择 6*sigma+1）
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1  # 确保核大小为奇数

    # 创建高斯核
    kernel = gaussian_kernel(kernel_size, sigma, 1)

    # 对直方图两端进行填充
    pad_size = kernel_size // 2
    padded_hist = F.pad(hist.unsqueeze(0).unsqueeze(0).float(), (pad_size, pad_size), mode="replicate")

    # 使用卷积实现高斯平滑
    smoothed_hist = F.conv1d(
        padded_hist,  # 转换为 [1, 1, bins + 2*pad_size]
        kernel,  # 转换为 [1, 1, kernel_size]
        padding=0
    ).squeeze()

    return smoothed_hist

def smooth_histogram(hist, sigma=3, window_size=1):
    """多步平滑直方图"""
    # 使用移动平均滤波器进行初步平滑
    moving_avg_hist = np.convolve(hist, np.ones(window_size)/window_size, mode='same')
    
    # 使用高斯滤波器进行最终平滑
    moving_avg_hist = torch.tensor(moving_avg_hist)
    smoothed_hist = smooth_histogram_gaussian(moving_avg_hist, sigma=sigma)
    
    return smoothed_hist

def compute_histogram_slope(hist): 
    """
    计算直方图的斜率。

    参数:
        hist (torch.Tensor): 输入直方图，形状为 [bins]。

    返回:
        slope (torch.Tensor): 直方图的斜率，形状为 [bins - 1]。
    """
    # 检查输入是否为一维张量
    if len(hist.shape) != 1:
        raise ValueError("Input histogram must be a 1D tensor.")

    # 计算斜率（差分）
    slope = hist[1:] - hist[:-1]

    return slope

def otsu_threshold(image):
    """
    输入:
        image: numpy array, shape (H, W), 单通道灰度图像
    输出:
        threshold: int, 最佳阈值
        variances: numpy array, shape (256,), 每个阈值对应的类间方差
    """
    assert len(image.shape) == 2, "输入图像必须是单通道（形状为 HxW）"

    # 统计直方图：每个灰度级的像素数量
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    total_pixels = image.size

    # 总灰度和
    sum_total = 0.0
    for i in range(256):
        sum_total += i * hist[i]

    # 初始化变量
    weight_background = 0.0
    sum_background = 0.0
    variances = np.zeros(256)

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue

        sum_background += t * hist[t]
        mean_background = sum_background / weight_background

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_foreground = sum_total - sum_background
        mean_foreground = sum_foreground / weight_foreground

        # 类间方差
        var_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        variances[t] = var_between

    best_threshold = np.argmax(variances)

    return int(best_threshold), variances
 
def robust_min_max(tensor, threshold=0.1, percentile=0.1):
    """
    Args:
        tensor (torch.Tensor): 输入的任意形状的 tensor
        threshold (float): 判断是否有效的阈值，只有大于该值的元素才被保留
        percentile (float): 百分比，用于选取 top 和 bottom 的比例，默认为 10%

    Returns:
        robust_min (float): 鲁棒最小值
        robust_max (float): 鲁棒最大值
    """
    # 1. 展平 tensor 并筛选出大于 threshold 的元素
    flat_tensor = tensor.view(-1)
    valid_elements = flat_tensor[flat_tensor > threshold]

    if len(valid_elements) == 0:
        return 0, 1e-8

    # 2. 排序
    sorted_elements = torch.sort(valid_elements).values

    # 3. 计算 10% 的数量，并向上取整
    num_elements = len(sorted_elements)
    k = max(1, int(num_elements * percentile + 0.5))  # 四舍五入并至少取1个

    # 4. 取前10%和后10%，并计算均值
    bottom_k = sorted_elements[:k]
    top_k = sorted_elements[-k:]

    robust_min = bottom_k.mean().item()
    robust_max = top_k.mean().item()

    return robust_min, robust_max

def dense_crf(logit_map, image, iter_num=5, bi_sdims=3, bi_schan=10, bi_compat=1, gaussian_sdims=3, gaussian_compat=1):
    """
    使用 DenseCRF 对前景 logit 图进行后处理优化。
    
    参数:
        logit_map: (H, W) float32，前景概率图（值在 [0,1]）
        image: (H, W, 3) 或 (H, W) float32，原始图像（归一化到 [0,1]）
        iter_num: int，CRF 推理次数
    
    返回:
        mask: (H, W) uint8，优化后的二值 mask（0: 背景, 1: 前景）
    """
    H, W = logit_map.shape

    # 重新归一化到 0-255 范围内
    min_val = image.min()
    max_val = image.max()
    image_normal = ((image - min_val) / (max_val - min_val)* 255).astype(np.uint8)
    # 如果输入是单通道图像，复制为三通道
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_normal = np.repeat(image_normal[..., np.newaxis], 3, axis=-1)

    # 构建 softmax 概率图：(2, H, W)
    prob_map = np.stack([1.0 - logit_map, logit_map], axis=0)  # 0: background, 1: foreground

    # Step 1: 构造一元势函数
    unary = unary_from_softmax(prob_map)  # shape: (2, H*W)

    # Step 2: 创建 DenseCRF 模型
    d = dcrf.DenseCRF2D(W, H, 2) 

    # Step 3: 设置一元势
    d.setUnaryEnergy(unary)

    # Step 4: 添加成对势（双边滤波：颜色 + 位置相似的像素倾向于同类别）
    feats = create_pairwise_bilateral(sdims=(bi_sdims, bi_sdims), schan=(bi_schan, bi_schan, bi_schan), img=image_normal, chdim=2)
    d.addPairwiseEnergy(feats, compat=bi_compat)
    # print(feats.shape)

    # Step 5: 添加空间平滑项（位置相近的像素倾向于同类别）
    feats_gaussian = create_pairwise_gaussian(sdims=(gaussian_sdims, gaussian_sdims), shape=(H, W))
    d.addPairwiseEnergy(feats_gaussian, compat=gaussian_compat)
    # print(feats_gaussian)

    # Step 6: 推理优化
    Q = d.inference(iter_num)
    Q = np.argmax(Q, axis=0).reshape((H, W))  # shape: (H, W)

    return Q.astype(np.uint8)

def hist_mapping(hist, alpha=1.):
    return 1 - torch.exp(-alpha*hist)

def filter_mask_by_points(mask: torch.Tensor, points: torch.Tensor, kernel_size=7) -> torch.Tensor:
    """
    根据点标签过滤 mask, 保留与点标签相重合的 mask 区域。
    
    参数:
        mask (torch.Tensor): 形状为 (H, W) 的二值张量，表示生成的 mask 标签。
        points (torch.Tensor): 形状为 (H, W) 的张量。
    
    返回:
        filtered_mask (torch.Tensor): 过滤后的 mask, 形状为 (H, W)。
    """
    # 扩展 mask 1 个像素，一边提高容忍性
    points = F.max_pool2d(points.float().unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze(0).squeeze(0)
    # 将 mask 转换为 numpy 数组以使用 scipy 的连通区域标记功能
    mask_np = mask.cpu().numpy()
    points_np = points.cpu().numpy()
    # mask_np = mask
    # points_np = points
    # 找到 mask 中的所有连通区域
    labeled_array, num_features = ndlabel(mask_np, structure=np.ones((3, 3)))
    # 初始化一个全零的数组用于存储过滤后的 mask
    filtered_mask = np.zeros_like(mask_np, dtype=np.uint8)
    # 遍历每个连通区域
    for i in range(1, num_features + 1):
        # 获取当前连通区域的布尔掩码
        region_mask = (labeled_array == i)
        
        # 检查是否有任何点标签落在当前连通区域内
        sum_ = np.sum(region_mask[points_np > 0.9])

        if sum_ > 0.:
            filtered_mask = filtered_mask + mask_np * region_mask
    
    # 将过滤后的 mask 转换回 torch 张量
    filtered_mask_tensor = torch.tensor(filtered_mask, dtype=torch.uint8, device=mask.device)
    
    return filtered_mask_tensor

def mapping_4_crf(tensor, grad_intensity, ratio=0.5, max_val=0.5001, min_val=0.4999):
    output = torch.zeros_like(tensor)

    tensor = torch.round(tensor * 10000) / 10000
    unique_val = torch.unique(tensor)   # 默认返回排序后得结果
    if unique_val.shape[0] <= 2:  # 确保至少有 2 个值
        max_val, min_val = 0.5001, 0.4999
    max_min_num = int(ratio * unique_val.shape[0])
    max_min_num = max_min_num if max_min_num >= 1 else 1
    
    maximum_mask = torch.isin(tensor, unique_val[-max_min_num:])
    grad_intensity_area = grad_intensity * maximum_mask
    area_min, area_max = grad_intensity_area[grad_intensity_area > 0.].min(), grad_intensity_area.max()
    grad_intensity_area = (grad_intensity_area-area_min) * (1-max_val)/(area_max - area_min) + max_val
    output = torch.where((grad_intensity_area > output)*maximum_mask, grad_intensity_area, output)

    minimum_mask = torch.isin(tensor, unique_val[:max_min_num])
    grad_intensity_area = grad_intensity * minimum_mask
    area_min, area_max = grad_intensity_area[grad_intensity_area > 0.].min(), grad_intensity_area.max()
    grad_intensity_area = (grad_intensity_area-area_min) * min_val/(area_max-area_min)
    output = torch.where((grad_intensity_area > output)*minimum_mask, grad_intensity_area, output)
    
    if unique_val.shape[0] <= 2:  # 确保至少有 2 个值
        return output

    medium_mask = torch.isin(tensor, unique_val[max_min_num:-max_min_num])
    grad_intensity_area = grad_intensity * medium_mask
    area_min, area_max = grad_intensity_area[grad_intensity_area > 0.].min(), grad_intensity_area.max()
    grad_intensity_area = (grad_intensity_area-area_min) * (max_val-min_val)/(area_max-area_min) + min_val
    output = torch.where((grad_intensity_area > output)*medium_mask, grad_intensity_area, output)

    return output

def mapping_4_crf_v2(tensor, grad_intensity, ratio=0.5):
    output = torch.zeros_like(tensor)

    max_val, min_val=0.5001, 0.4999
    tensor = torch.round(tensor * 10000) / 10000
    unique_val = torch.unique(tensor)   # 默认返回排序后得结果
    if unique_val.shape[0] <= 2:  # 确保至少有 2 个值
        max_val, min_val = 0.5001, 0.4999
    max_min_num = int(ratio * unique_val.shape[0])
    max_min_num = max_min_num if max_min_num >= 1 else 1
    
    maximum_mask = torch.isin(tensor, unique_val[-max_min_num:])
    grad_intensity_area = grad_intensity * maximum_mask
    area_min, area_max = grad_intensity_area[grad_intensity_area > 0.].min(), grad_intensity_area.max()
    max_val = 0.5 + area_min * 0.5
    grad_intensity_area = (grad_intensity_area-area_min) * (1-max_val)/(area_max - area_min) + max_val
    output = torch.where((grad_intensity_area > output)*maximum_mask, grad_intensity_area, output)
    if unique_val.shape[0] <= 2:  # 确保至少有 2 个值
        return output
    
    min_val = 0.5 - area_min * 0.5
    medium_mask = torch.isin(tensor, unique_val[max_min_num:-max_min_num])
    grad_intensity_area = grad_intensity * medium_mask
    area_min, area_max = grad_intensity_area[grad_intensity_area > 0.].min(), grad_intensity_area.max()
    grad_intensity_area = grad_intensity_area * (max_val-min_val)/(area_max-area_min) + min_val
    output = torch.where((grad_intensity_area > output)*medium_mask, grad_intensity_area, output)

    return output

def mapping_4_crf_v3(tensor, grad_intensity, ratio=0.5, max_val=0.5001, min_val=0.4999):
    output = torch.zeros_like(tensor)

    tensor = torch.round(tensor * 10000) / 10000
    unique_val = torch.unique(tensor)   # 默认返回排序后得结果
    max_min_num = int(ratio * unique_val.shape[0])
    max_min_num = max_min_num if max_min_num >= 1 else 1
    
    maximum_mask = torch.isin(tensor, unique_val[-max_min_num:])
    grad_intensity_area = grad_intensity * maximum_mask
    area_min, area_max = grad_intensity_area[grad_intensity_area > 0.].min(), grad_intensity_area.max()
    grad_intensity_area = (grad_intensity_area-area_min) * (1-max_val)/(area_max - area_min) + max_val
    output = torch.where((grad_intensity_area > output)*maximum_mask, grad_intensity_area, output)

    minimum_mask = torch.isin(tensor, unique_val[:max_min_num])
    grad_intensity_area = grad_intensity * minimum_mask
    area_min, area_max = grad_intensity_area[grad_intensity_area > 0.].min(), grad_intensity_area.max()
    grad_intensity_area = (grad_intensity_area-area_min) * min_val/(area_max-area_min)
    output = torch.where((grad_intensity_area > output)*minimum_mask, grad_intensity_area, output)
    
    if unique_val.shape[0] <= 2:  # 确保至少有 2 个值
        return output

    max_val = 0.5
    medium_mask = torch.isin(tensor, unique_val[max_min_num:-max_min_num])
    grad_intensity_area = grad_intensity * medium_mask
    area_min, area_max = grad_intensity_area[grad_intensity_area > 0.].min(), grad_intensity_area.max()
    grad_intensity_area = (grad_intensity_area-area_min) * (max_val-min_val)/(area_max-area_min) + min_val
    output = torch.where((grad_intensity_area > output)*medium_mask, grad_intensity_area, output)

    return output

def mapping_4_crf_v4(tensor, grad_intensity, ratio=0.5, node_list=[0.8, 0.6, 0.4, 0.2]):
    output = torch.zeros_like(tensor)

    tensor = torch.round(tensor * 10000) / 10000
    unique_val = torch.unique(tensor)   # 默认返回排序后得结果
    max_min_num = int(ratio * unique_val.shape[0])
    max_min_num = max_min_num if max_min_num >= 1 else 1
    
    max_val = node_list[0]
    maximum_mask = torch.isin(tensor, unique_val[-max_min_num:])
    grad_intensity_area = grad_intensity * maximum_mask
    area_min, area_max = grad_intensity_area[grad_intensity_area > 0.].min(), grad_intensity_area.max()
    grad_intensity_area = (grad_intensity_area-area_min) * (1-max_val)/(area_max - area_min) + max_val
    output = torch.where((grad_intensity_area > output)*maximum_mask, grad_intensity_area, output)

    min_val = node_list[3]
    minimum_mask = torch.isin(tensor, unique_val[:max_min_num])
    grad_intensity_area = grad_intensity * minimum_mask
    area_min, area_max = grad_intensity_area[grad_intensity_area > 0.].min(), grad_intensity_area.max()
    grad_intensity_area = (grad_intensity_area-area_min) * min_val/(area_max-area_min)
    output = torch.where((grad_intensity_area > output)*minimum_mask, grad_intensity_area, output)
    
    if unique_val.shape[0] <= 2:  # 确保至少有 2 个值
        return output

    max_val, min_val = node_list[1], node_list[2]
    medium_mask = torch.isin(tensor, unique_val[max_min_num:-max_min_num])
    grad_intensity_area = grad_intensity * medium_mask
    area_min, area_max = grad_intensity_area[grad_intensity_area > 0.].min(), grad_intensity_area.max()
    grad_intensity_area = (grad_intensity_area-area_min) * (max_val-min_val)/(area_max-area_min) + min_val
    output = torch.where((grad_intensity_area > output)*medium_mask, grad_intensity_area, output)

    return output

def mapping_4_crf_v5(tensor, grad_intensity, ratio=0.5, node_list=[0.7, 0.5]):
    output = torch.zeros_like(tensor)

    tensor = torch.round(tensor * 10000) / 10000
    unique_val = torch.unique(tensor)   # 默认返回排序后得结果
    max_min_num = int(ratio * unique_val.shape[0])
    max_min_num = max_min_num if max_min_num >= 1 else 1
    
    max_val = node_list[0]
    maximum_mask = torch.isin(tensor, unique_val[-max_min_num:])
    grad_intensity_area = grad_intensity * maximum_mask
    area_min, area_max = grad_intensity_area[grad_intensity_area > 0.].min(), grad_intensity_area.max()
    grad_intensity_area = (grad_intensity_area-area_min) * (1-max_val)/(area_max - area_min) + max_val
    output = torch.where((grad_intensity_area > output)*maximum_mask, grad_intensity_area, output)

    min_val = node_list[1]
    minimum_mask = ~maximum_mask
    grad_intensity_area = grad_intensity * minimum_mask
    area_min, area_max = grad_intensity_area[grad_intensity_area > 0.].min(), grad_intensity_area.max()
    grad_intensity_area = (grad_intensity_area-area_min) * min_val/(area_max-area_min)
    output = torch.where((grad_intensity_area > output)*minimum_mask, grad_intensity_area, output)

    return output