import torch
import torch.nn.functional as F
import numpy as np

from utils.utils import iou_score, gaussian_blurring_2D

def proper_region(pred, c1, c2):
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
    initial_size = 32
    half_size = initial_size // 2
    pred_ = F.pad(pred, [half_size, half_size, half_size, half_size], value=0)
    s1 = c1
    e1 = c1 + initial_size
    s2 = c2
    e2 = c2 + initial_size
    mini_size = 2
    extend_size = 3
    # 合适上边界
    for i in range(half_size, mini_size//2, -1):
        s1 = c1 + half_size - i
        if torch.max(pred_[s1, s2:e2]) > 0.1:
            break
    # 下边界
    for i in range(half_size, mini_size//2, -1):
        e1 = c1 + half_size + i
        if torch.max(pred_[e1, s2:e2]) > 0.1:
            break
    # 左边界
    for i in range(half_size, mini_size//2, -1):
        s2 = c2 + half_size - i
        if torch.max(pred_[s1:e1, s2]) > 0.1:
            break
    # 右边界
    for i in range(half_size, mini_size//2, -1):
        e2 = c2 + half_size + i
        if torch.max(pred_[s1:e1, e2]) > 0.1:
            break
    
    s1 = s1 - half_size - extend_size if s1 - half_size - extend_size > 1 else 1
    e1 = e1 - half_size + extend_size if e1 - half_size + extend_size < pred.shape[0] - 2 else pred.shape[0] - 2
    s2 = s2 - half_size - extend_size if s2 - half_size - extend_size > 1 else 1
    e2 = e2 - half_size + extend_size if e2 - half_size + extend_size < pred.shape[1] - 2 else pred.shape[1] - 2
    return (s1, e1, s2, e2)


def examine_iou(final_target, pesudo_label, iou_treshold=0.5):
    """
    最终伪标签与上轮伪标签的iou，并返回结果。
    final_target (torch.Tensor): (H,W)模型输出的伪标签。
    pesudo_label (torch.Tensor): (H,W)上轮的伪标签。
    iou_treshold (float): iou阈值，默认为0.5。
    """
    if torch.max(pesudo_label) < 0.1:
        return final_target
    iou = iou_score(final_target.numpy() > 0.1, pesudo_label.numpy() > 0.1)
    # print(iou)
    if iou < iou_treshold:
        return pesudo_label
    else:
        return final_target
    

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


