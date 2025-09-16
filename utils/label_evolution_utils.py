import torch
import torch.nn.functional as F
import numpy as np

from utils.utils import iou_score, gaussian_blurring_2D, gaussian_kernel

def proper_region(pred, c1, c2, extend_factor=0.5):
    """
    由训练过的模型的预测和点标签的坐标，得到一个合适的区域。
    参数:
        pred (torch.Tensor): 形状为 [H, W]。
    输出：
        s1 (int): 区域的起始高度索引。
        e1 (int): 区域的结束高度索引。
        s2 (int): 区域的起始宽度索引。
        e2 (int): 区域的结束宽度索引。
    """
    initial_size = 64
    half_size = initial_size // 2
    pred_ = F.pad(pred, [half_size, half_size, half_size, half_size], value=0)
    s1 = c1
    e1 = c1 + initial_size
    s2 = c2
    e2 = c2 + initial_size
    mini_size = 6
    # 合适上边界
    for i in range(mini_size//2, half_size):
        s1 = c1 + half_size - i
        if torch.sum(pred_[s1, s2:e2]) < 1.:
            break
    # 下边界
    for i in range(mini_size//2, half_size):
        e1 = c1 + half_size + i + 1
        if torch.sum(pred_[e1, s2:e2])  < 1:
            break
    # 左边界
    for i in range(mini_size//2, half_size):
        s2 = c2 + half_size - i
        if torch.sum(pred_[s1:e1, s2])  < 1:
            break
    # 右边界
    for i in range(mini_size//2, half_size):
        e2 = c2 + half_size + i + 1
        if torch.sum(pred_[s1:e1, e2])  < 1:
            break

    s1, e1, s2, e2 = s1 - half_size, e1 - half_size, s2 - half_size, e2 - half_size
    s1, e1 = max(s1, 1), min(e1, pred.shape[0] - 2)
    s2, e2 = max(s2, 1), min(e2, pred.shape[1] - 2)
    
    s1_, e1_ = s1 - int((e1 - s1) * extend_factor / 2), e1 + int((e1 - s1) * extend_factor / 2)
    s2_, e2_ = s2 - int((e2 - s2) * extend_factor / 2), e2 + int((e2 - s2) * extend_factor / 2)
    s1_ = s1_ if s1_ > 1 else 1
    e1_ = e1_ if e1_ < pred.shape[0] - 2 else pred.shape[0] - 2
    s2_ = s2_ if s2_ > 1 else 1
    e2_ = e2_ if e2_ < pred.shape[1] - 2 else pred.shape[1] - 2
    # print(c1, c2, s1, e1, s2, e2, s1_, e1_, s2_, e2_)
    
    return (int(s1_), int(e1_), int(s2_), int(e2_))

def examine_iou(final_target, pesudo_label, image, iou_treshold=0.5):
    """
    最终伪标签与上轮伪标签的iou，并返回结果。
    final_target (torch.Tensor): (H,W)模型输出的伪标签。
    pesudo_label (torch.Tensor): (H,W)上轮的伪标签。
    image(torch.Tensor): (H,W)上轮的伪标签。
    iou_treshold (float): iou阈值，默认为0.5。
    """
    if (final_target * pesudo_label).float().sum() >= 4:
        iou = iou_score(final_target.numpy() > 0.1, pesudo_label.numpy() > 0.1)
        # print(iou)
        if iou < iou_treshold:
            return pesudo_label
        else:
            return final_target
    elif final_target.float().sum() >= 4:
        return final_target
    elif pesudo_label.float().sum() >= 4: 
        return pesudo_label
    else :
        return torch.zeros_like(pesudo_label)
    
# def advice_region(coors, coors2, target_mask, pesudo_label, image, iou_treshold=0.5):
#     target_mask_ = target_mask[coors[0]:coors[1], coors[2]:coors[3]]
#     pesudo_label_ = pesudo_label[coors[0]:coors[1], coors[2]:coors[3]]
#     image_ = image[coors[0]:coors[1], coors[2]:coors[3]]
#     advice = examine_iou(target_mask_, pesudo_label_, image_, iou_treshold)
#     return advice


def expand_and_contract_mask(mask, d1, d2):
    """
    对目标mask的边缘进行向外和向内扩展。
    
    参数:
        mask (torch.Tensor): 输入的目标mask，形状为 [1, 1, H, W]。
        d1 (int): 向外扩展的像素数。
        d2 (int): 向内收缩的像素数。
    
    返回:
        torch.Tensor: 处理后的mask，形状与输入相同，取值为0或1。
    """
    
    # 使用最大池化实现向外扩展
    kernel_size_d1 = 2 * d1 + 1
    expanded_mask = F.max_pool2d(mask.float(), kernel_size=kernel_size_d1, stride=1, padding=d1)
    
    # 使用腐蚀操作（最小池化）实现向内收缩
    kernel_size_d2 = 2 * d2 + 1
    contracted_mask = -F.max_pool2d(-mask.float(), kernel_size=kernel_size_d2, stride=1, padding=d2)

    # 对结果取高斯模糊，即不产生锐利的mask，而是宽容度更高的
    kernel_size = min(d1-1, d2-1)*2 + 1
    gaussian_kernel_1d = gaussian_kernel(kernel_size, 2)
    gaussian_kernel_ = torch.outer(gaussian_kernel_1d, gaussian_kernel_1d)
    gaussian_kernel_ = gaussian_kernel_.expand(1, 1, kernel_size, kernel_size)
    result_mask = F.conv2d(expanded_mask, gaussian_kernel_, padding=gaussian_kernel_.shape[-1]//2)

    # 取两者的交集：向外扩展的部分与向内收缩的部分
    result_mask = result_mask * (contracted_mask < 1.0)
    # result_mask = (result_mask > 0.5).float()

    return result_mask


def smooth_and_scale_mask(mask, a=0.1, b=0.9, sigma=None, kernel_size=None):
    """
    Args:
        mask (torch.Tensor): shape [H, W], 值接近 0.0 或 1.0。
        a (float): 输出的最小值
        b (float): 输出的最大值
        sigma (float): 高斯模糊的标准差
    
    Returns:
        torch.Tensor: 处理后的 mask，shape [H, W]，值在 [a, b] 范围内
    """
    # 确保输入是 float 类型
    mask = mask.float()
    if sigma is not None:
        # mask_ = dilate_mask(mask, 1)
        mask_ = mask

        # 高斯模糊
        mask_smooth = gaussian_blurring_2D(mask_, kernel_size=kernel_size, sigma=sigma)
        mask = mask_smooth  # [H, W]

    # 线性变换到 [a, b]
    x_min, x_max = mask.min(), mask.max()

    # 使用线性映射：x' = a + (x - x_min) * (b - a) / (x_max - x_min)
    x_scaled = a + (mask - x_min) * (b - a) / (x_max - x_min + 1e-8)  # 加上小数防止除零

    return x_scaled


def create_fading_tensor(H, W):
    """
    生成一个形状为 [H, W] 的 PyTorch 张量，
    值从中心 (1.0) 向四周逐渐衰减到 0。
    使用归一化的欧氏距离实现。
    """
    # 创建坐标网格
    y = torch.linspace(0, H - 1, H)
    x = torch.linspace(0, W - 1, W)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # 注意：PyTorch 1.10+ 推荐使用 indexing='ij'

    # 中心点坐标
    center_y, center_x = (H - 1) / 2, (W - 1) / 2

    # 计算每个点到中心的欧氏距离
    dist = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)

    # 归一化距离到 [0, max_dist] 范围，然后反向映射到 [1, 0]
    max_dist = torch.sqrt(torch.tensor(center_x**2 + center_y**2)) / 2  # 最大可能距离（从中心到角落）
    normalized_dist = dist / max_dist

    # 衰减函数：1 - distance，确保范围在 [0, 1]
    tensor = 1.0 - normalized_dist.clamp(0, 1)

    return tensor


def fusion_tm_dl(target, pred, alpha=0.5, beta=0.75, sigma=0.9):
    """
    融合固定算法和深度学习模型所产生的伪标签。
    参数:
    target (torch.Tensor): 算法输出的伪标签。
    pred (torch.Tensor): 模型输出的预测， 仅包含。
    """
    aux_pred = smooth_and_scale_mask(pred, alpha, 1.0)
    aux_target = smooth_and_scale_mask(target, beta, 1.0)
    fusion = aux_pred * aux_target
    min_val, max_val = fusion.min(), fusion.max()
    return (fusion - min_val) / (max_val - min_val + 1e-8)


def fusion_tm_dl_v2(target, pred):
    """
    融合固定算法和深度学习模型所产生的伪标签。
    原则是通过遍历target和pred两个伪标签候选的决策权重，形成不pred不一致且与target中为1的区域尽量不一致的新的伪标签。
    参数:
    target (torch.Tensor): 算法输出的伪标签。
    pred (torch.Tensor): 模型输出的预测， 仅包含。
    """
    target_mask = (target >= 1.).float()
    pred_mask = (pred > 0.1).float()
    if torch.max(pred_mask) <= 0.1:
        return target_mask
    IoU = iou_score(target_mask.numpy(), pred_mask.numpy())
    if IoU > 0.9:
        return target_mask
    
    # 四舍五入到小数点后 4 位再取唯一值
    unique_target = torch.unique(torch.round(target * 10000) / 10000)   # 默认返回排序后得结果
    if unique_target.shape[0] <= 2:  # 确保至少有 2 个值
        target_lower_limit = unique_target[1] * 0.8
    else:
        target_lower_limit = 2 * unique_target[1] - unique_target[2]
        target_lower_limit = target_lower_limit if target_lower_limit > 0 else 0
    unique_target[0] = target_lower_limit
    # print('unique_target: ', unique_target)

    #
    def fusion_score(target_mask, pred_mask, filtered_mask):
        """
        通过计算filtered_mask是否达到与pred不同，与target尽量不同得效果
        Returns:
            fusion_score, float
        """
        target_only = (1-pred_mask) * target_mask
        pred_only = (1-target_mask) * pred_mask
        inter_area = target_mask * pred_mask
        target_only_iou = iou_score(target_only.numpy(), filtered_mask.numpy())
        pred_only_iou = iou_score(pred_only.numpy(), filtered_mask.numpy())
        inter_area_iou = iou_score(inter_area.numpy(), filtered_mask.numpy())

        score = 0
        if torch.min(pred_mask[target_mask == 1]) == 1:
            score -= 1.0
        elif torch.min(target_mask[pred_mask == 1]) == 1:
            score -= 0.5
        elif target_only_iou == 0:
            score -= 1.0
        elif pred_only_iou == 0:
            score -= 0.5

        score += 0.3 * target_only_iou + 0.3 * pred_only_iou + 0.4 * inter_area_iou
        # print(target_only_iou, pred_only_iou, inter_area_iou, score)
        return score

    scores = []
    filtered_areas = []
    target_ = torch.clamp_min(target, target_lower_limit)
    for i in range(1, unique_target.shape[0]):
        for j in range(i, unique_target.shape[0]):
            pred_lower_limit = (unique_target[i] + unique_target[i-1])/(2 * unique_target[j])
            pred_ = pred_lower_limit + pred_mask * (1 - pred_lower_limit)
        
            fusion = pred_ * target_

            unique_fusion = torch.unique(torch.round(fusion * 10000) / 10000)
            # print('unique_fusion: ',unique_fusion)

            for k in range(1, unique_fusion.shape[0]):
                filtered_area = (fusion >= unique_fusion[k]).float()
                score = fusion_score(target_mask, pred_mask, filtered_area)
                scores.append(score)
                filtered_areas.append(filtered_area)
    # print(scores)
    
    max_score_idx = np.argmax(scores)
    
    return filtered_areas[max_score_idx]


