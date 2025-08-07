import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.utils import extract_local_windows

torch.set_printoptions(
    precision=4,
    threshold=1024,
    edgeitems=320,
    linewidth=1024,
    sci_mode=True,
    profile="compact",
)


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
    eroded = F.conv2d(mask, weight, padding=kernel_size // 2)
    eroded = (eroded == kernel_size * kernel_size).float().squeeze()
    return eroded

def get_verified_region(seed_mask, dilate_iter=1, erode_iter=1, return_erode_mask=False):
    seed_mask = seed_mask.float()
    dilated = dilate_mask(seed_mask, d=dilate_iter)
    eroded = erode_mask(seed_mask, d=erode_iter)
    verified = dilated - eroded  # 只保留膨胀但未被保留的边界区域
    if return_erode_mask:
        return verified.clamp(0, 1).float(), eroded
    return verified.clamp(0, 1).float()


# def estimate_bg_fg(tensor, mask, window_size):
#     """
#     tensor: shape [H, W]
#     mask: shape [H, W], boolen mask
#     返回：
#         bg_values: shape [N]
#         fg_values: shape [N]
#     """
#     tensor_window = extract_local_windows(tensor, window_size)  # [H, W, 25]
#     mask_window = extract_local_windows(mask, window_size)  # [H, W, 25]

#     tensor_window_bg = tensor_window * (mask_window == 0)
#     tensor_window_fg = tensor_window * (mask_window == 1)  # [H, W, 25]
#     fg_cnt = torch.sum(mask_window, dim=[-2, -1])  # [H, W]

#     # # 排序(暂时不用中值了，因为觉得不会有很多极端值来影响)
#     # H, W, _, _ = tensor_window.shape
#     # flat = tensor_window.view(H, W, -1)  # [H, W, window_size^2]
#     # sorted_vals, _ = torch.sort(flat, dim=1)

#     bg_mean = torch.sum(tensor_window_bg, dim=[-2, -1]) / (window_size * window_size - fg_cnt).clamp(min=1)  # [H, W]
#     fg_mean = torch.sum(tensor_window_fg, dim=[-2, -1]) / fg_cnt.clamp(min=1)  # [H, W]

#     return bg_mean, fg_mean


def get_gaussian_kernel(window_size, sigma=None):
    """
    生成一个 window_size x window_size 的高斯权重矩阵
    """
    if sigma is None:
        sigma = window_size / 3  # 自动设置sigma
    coords = torch.arange(window_size) - window_size // 2
    x, y = torch.meshgrid(coords, coords)
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # 归一化
    return kernel  # shape: [window_size, window_size]


def _estimate_bg_fg(tensor, fg_mask, bg_mask, window_size, target_area_mask=None, target_area_weight=0.5):
    """
    tensor: shape [H, W]
    fg_mask: shape [H, W], bool mask (True 表示前景)
    bg_mask: shape [H, W], bool mask (True 表示背景)
    target_area_mask: shape [H, W], bool mask (None 表示目标区域)
    返回：
        bg_values: shape [H, W]
        fg_values: shape [H, W]
    """
    device = tensor.device
    # 1. 提取局部窗口
    tensor_window = extract_local_windows(tensor, window_size)  # shape [H, W, win, win]
    fg_mask_window = extract_local_windows(fg_mask.float(), window_size).bool()  # shape [H, W, win, win]
    bg_mask_window = extract_local_windows(bg_mask.float(), window_size).bool()  # shape [H, W, win, win]

    # 2. 构建高斯权重矩阵
    gaussian_weights = get_gaussian_kernel(window_size, sigma=2).to(device)
    gaussian_weights = gaussian_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, win, win]

    # 3. 分离背景和前景部分
    tensor_window_bg = tensor_window * bg_mask_window
    tensor_window_fg = tensor_window * fg_mask_window

    # 4. 加权求和
    weighted_bg = tensor_window_bg * gaussian_weights 
    weighted_fg = tensor_window_fg * gaussian_weights

    # 5. 求加权总和 & 总权重（避免除零）
    bg_weighted_sum = weighted_bg.sum(dim=[-2, -1])
    fg_weighted_sum = weighted_fg.sum(dim=[-2, -1])

    bg_total_weight = bg_mask_window.float().mul(gaussian_weights).sum(dim=[-2, -1]).clamp(min=1e-3)
    fg_total_weight = fg_mask_window.float().mul(gaussian_weights).sum(dim=[-2, -1]).clamp(min=1e-3)

    # 6. 计算加权均值
    bg_mean = bg_weighted_sum / bg_total_weight
    fg_mean = fg_weighted_sum / fg_total_weight

    # 显示图像
    plt.figure(figsize=(30, 5))
    plt.subplot(161), plt.imshow(tensor, cmap='gray', vmax=1.0, vmin=0.0)
    plt.subplot(162), plt.imshow(bg_weighted_sum, cmap='gray')
    plt.subplot(163), plt.imshow(fg_weighted_sum, cmap='gray')
    plt.subplot(164), plt.imshow(bg_total_weight, cmap='gray')
    plt.subplot(165), plt.imshow(fg_total_weight, cmap='gray')
    plt.subplot(166), plt.imshow(target_area_mask, cmap='gray')
    print(torch.max(bg_mean), torch.max(fg_mean))
    plt.show()

    return bg_mean, fg_mean


def estimate_bg_fg(tensor, fg_mask, bg_mask, window_size, target_area_mask=None, target_area_weight=0.5):
    """
    tensor: shape [H, W]
    fg_mask: shape [H, W], bool mask (True 表示前景)
    bg_mask: shape [H, W], bool mask (True 表示背景)
    target_area_mask: shape [H, W], bool mask (None 表示目标区域 = 原前景 - 内部前景)
    target_area_weight: float, 目标区域使用的权重比例
    返回：
        bg_values: shape [H, W]
        fg_values: shape [H, W]
    """
    device = tensor.device
    dtype = tensor.dtype

    # 1. 提取局部窗口
    tensor_window = extract_local_windows(tensor, window_size)  # shape [H, W, win, win]
    fg_mask_window = extract_local_windows(fg_mask.float(), window_size).bool()  # shape [H, W, win, win]
    bg_mask_window = extract_local_windows(bg_mask.float(), window_size).bool()  # shape [H, W, win, win]

    target_area_mask_ = extract_local_windows(target_area_mask.float(), window_size).bool()

    # 构建不同的权重掩码
    gaussian_weights = get_gaussian_kernel(window_size, sigma=2).to(device=device, dtype=dtype)
    gaussian_weights = gaussian_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, win, win]

    # 构造 target_area 的权重：降低一部分
    target_area_weight_matrix = torch.where(
        target_area_mask_,
        torch.tensor(target_area_weight, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype)
    )

    final_weights = gaussian_weights * target_area_weight_matrix

    # 分离背景和前景部分
    tensor_window_bg = tensor_window * (bg_mask_window + target_area_mask_)
    tensor_window_fg = tensor_window * (fg_mask_window + target_area_mask_)

    # 加权求和
    weighted_bg = tensor_window_bg * final_weights 
    weighted_fg = tensor_window_fg * final_weights

    # 求加权总和 & 总权重（避免除零）
    bg_weighted_sum = weighted_bg.sum(dim=[-2, -1])
    fg_weighted_sum = weighted_fg.sum(dim=[-2, -1])

    bg_total_weight = (bg_mask_window + target_area_mask_).float().mul(final_weights).sum(dim=[-2, -1]).clamp(min=1e-3)
    fg_total_weight = (fg_mask_window + target_area_mask_).float().mul(final_weights).sum(dim=[-2, -1]).clamp(min=1e-3)

    # 计算加权均值
    bg_mean = bg_weighted_sum / bg_total_weight
    fg_mean = fg_weighted_sum / fg_total_weight

    # print(tensor[12:18,15:20])
    # print(fg_mean[12:18,15:20])
    # print(fg_total_weight[12:18,15:20])

    # # 显示图像
    # plt.figure(figsize=(30, 5))
    # plt.subplot(161), plt.imshow(tensor, cmap='gray', vmax=1.0, vmin=0.0)
    # plt.subplot(162), plt.imshow(bg_mean, cmap='gray')
    # plt.subplot(163), plt.imshow(fg_mean, cmap='gray')
    # plt.subplot(164), plt.imshow(bg_total_weight, cmap='gray')
    # plt.subplot(165), plt.imshow(fg_total_weight, cmap='gray')
    # plt.subplot(166), plt.imshow(target_area_mask, cmap='gray')
    # print(torch.max(bg_mean), torch.max(fg_mean))
    # plt.show()

    return bg_mean, fg_mean


def estimate_bg_fg_totalmean(tensor, fg_mask, bg_mask):
    """
    tensor: shape [H, W], 概率图
    fg_mask: shape [H, W], bool, True 表示前景像素
    bg_mask: shape [H, W], bool, True 表示背景像素
    返回:
        bg_mean: float, 背景像素的加权平均值
        fg_mean: float, 前景像素的加权平均值
    """
    # 计算前景均值
    fg_sum = tensor[fg_mask.bool()].sum()
    fg_count = fg_mask.sum().float()
    fg_mean = fg_sum / fg_count.clamp(min=1e-6)

    # 计算背景均值
    bg_sum = tensor[bg_mask.bool()].sum()
    bg_count = bg_mask.sum().float()
    bg_mean = bg_sum / bg_count.clamp(min=1e-6)

    return bg_mean, fg_mean


def alpha4diff_ratio(bg_mean, fg_mean, turning_point=0.1):
    """
    根据 bg_mean 和前景均值 fg_mean 的差值生成 alpha 权重。
    
    参数:
        bg_mean: 当前像素概率图 [H, W]
        fg_mean: 局部前景均值 [H, W]
        sharpness: 控制衰减速率的参数，越大越陡峭
        
    返回:
        alpha: 权重系数 [H, W]，范围 [0, 1]
    """
    diff = torch.abs(bg_mean - fg_mean)
    # alpha = torch.sigmoid(-sharpness * diff + 5)  # 调整偏移使中间为 0.5
    alpha = torch.sqrt(4*turning_point  * diff)
    alpha = (alpha - alpha.min())/(alpha.max() - alpha.min()) * 0.8
    return alpha


def combined_discrimination(prob, fg_mean, bg_mean, mask, return_score=False):
    """
    综合差值和比值判断是否是目标像素

    参数:
        prob: 当前像素值 [H, W]
        fg_mean: 局部前景均值 [H, W]
        bg_mean: 局部背景均值 [H, W]
        alpha: 差值项权重
        beta: 比值项权重

    返回:
        is_target: bool mask [H, W]
    """
    # 如果当前像素大于fg,或者小于bg，额外的处理
    is_fg = prob > fg_mean
    is_bg = prob < bg_mean
    # 差和比的权重
    alpha = alpha4diff_ratio(bg_mean, fg_mean, turning_point=0.4)
    # alpha = 0.9
    # 差值项
    diff_score = torch.abs(fg_mean - prob) - torch.abs(prob - bg_mean)
    # diff_score = torch.where(diff < 0, -torch.ones_like(diff), torch.ones_like(diff))
    # 比值项
    eps = 1e-3
    ratio = (fg_mean / (prob + eps)) - (prob / (bg_mean + eps))
    ratio_negative_mask = ratio < 0.
    ratio_max, ratio_min = ratio[ratio_negative_mask].max(), ratio[ratio_negative_mask].min()
    ratio_score = torch.where(ratio_negative_mask, -(ratio - ratio_min)/(ratio_max - ratio_min), ratio)
    # ratio_score = torch.where(ratio < 0, -torch.ones_like(ratio), torch.ones_like(ratio))
    # print(diff_score[15:25, 15:25])
    # print(ratio_score[15:25, 15:25])
    # print(alpha[15:25, 15:25])
    # with open("tensor_output.txt", "w") as f:
    #     print(alpha, file=f)
    #     print(diff_score, file=f)
    #     print(ratio_score, file=f)
    #     print(ratio, file=f)
    # 综合得分
    score = alpha * diff_score + (1-alpha)* ratio_score
    score = torch.where(is_fg, -torch.ones_like(score), score)
    score = torch.where(is_bg, torch.ones_like(score), score)
    if return_score:
        return score, (diff_score, ratio_score, alpha)
    return score


def refine_iter(prob_map, mask, iter_num=5, window_size=5):
    """
    迭代优化mask
    prob_map: shape [H, W], 扩展后的梯度图
    mask: shape [H, W], 当前的mask
    iter_num: 迭代次数
    window_size: 窗口大小
    返回：
        mask: shape [H, W], 优化后的mask
    """
    recent_mask = mask.clone()
    for i in range(0,iter_num):
        if i % 2 == 0:
            erode_iter = 0
            dilate_iter = 1
            verified_region, _recent_mask= get_verified_region(recent_mask, dilate_iter=dilate_iter, erode_iter=erode_iter, return_erode_mask=True)
        else:
            erode_iter = 1
            dilate_iter = 0
            verified_region_, _recent_mask = get_verified_region(_recent_mask, dilate_iter=dilate_iter, erode_iter=erode_iter, return_erode_mask=True)

        # 2. 计算背景和前景的均值
        bg_mean, fg_mean = estimate_bg_fg(prob_map, _recent_mask, ~recent_mask.bool(), window_size, verified_region, target_area_weight=0.1)
        # 2.5 计算总体均值
        bg_total_mean, fg_total_mean = estimate_bg_fg_totalmean(prob_map, _recent_mask, ~recent_mask.bool())
        # 2.6 计算总体均值权重
        alpha = (i+1) / iter_num * 0.3 +  0.2
        bg_mean = bg_mean * (1. - alpha) + bg_total_mean * alpha
        # fg_mean = fg_mean * (1. - alpha) + fg_total_mean * alpha
        # 3. 判别是否为目标
        contrast, scores = combined_discrimination(prob_map, fg_mean, bg_mean, verified_region, return_score=True)
        contrast[0,0] = 0.
        is_target = contrast < 0.
        # 4. 更新 mask
        recent_mask = verified_region * is_target.float() + _recent_mask
        # 显示图像
        fig, axes = plt.subplots(2, 4, figsize=(10, 20))  # 调整 figsize 以适应布局
    
        ax = axes[0, 0]
        ax.imshow(prob_map, cmap='gray', vmax=1.0, vmin=0.0)
        ax.set_title("prob_map")

        ax = axes[0, 1]
        ax.imshow(recent_mask * prob_map, cmap='gray', vmax=1.0, vmin=0.0)
        ax.set_title("recent_mask * prob_map")

        ax = axes[0, 2]
        ax.imshow(fg_mean, cmap='gray', vmax=1.0, vmin=0.0)
        ax.set_title("fg_mean")

        ax = axes[0, 3]
        ax.imshow(bg_mean, cmap='gray', vmax=1.0, vmin=0.0)
        ax.set_title("bg_mean")

        ax = axes[1, 0]
        ax.imshow(scores[2], cmap='gray', vmax=1.0, vmin=0.0)
        ax.set_title("coefficient")

        ax = axes[1, 1]
        ax.imshow(_recent_mask * prob_map, cmap='gray', vmax=1.0, vmin=0.0)
        ax.set_title("_recent_mask * prob_map")

        ax = axes[1, 2]
        ax.imshow(scores[0], cmap='gray', vmax=0.0)
        ax.set_title("diff_score")

        ax = axes[1, 3]
        ax.imshow(scores[1], cmap='gray', vmax=0.0)
        ax.set_title("ratio_score")

        # 调整子图间距
        plt.suptitle(f'i: {i}, alpah: {alpha}', fontsize=16)
        plt.tight_layout()
        plt.show()

    return recent_mask

def refine_iter_v2(prob_map, mask, iter_num=5, window_size=5):
    recent_mask = mask.clone()
    iter_num *= 3
    for i in range(0,iter_num):
        if i % 3 == 0:
            erode_iter = 0
            dilate_iter = 1
            verified_region, _recent_mask= get_verified_region(recent_mask, dilate_iter=dilate_iter, erode_iter=erode_iter, return_erode_mask=True)
        elif i % 3 == 2:
            erode_iter = 1
            dilate_iter = 0
            verified_region, _recent_mask = get_verified_region(recent_mask, dilate_iter=dilate_iter, erode_iter=erode_iter, return_erode_mask=True)
        
        # 1. 获取verified_region对应的周围区域
        regions = get_verified_region_around(verified_region, window_size)

        # 2.对一个区域施行otsu阈值算法
        otsu_threshold_ = otsu_threshold(verified_region_)
        