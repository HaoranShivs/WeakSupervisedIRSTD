import torch
from torch.nn.functional import conv2d, max_pool2d

import matplotlib.pyplot as plt
from collections import deque

from utils.utils import gaussian_kernel, extract_local_windows
from utils.refine import dilate_mask

E = 2.71828183


def min_pool2d(input, kernel_size, stride=None, padding=0):
    # Invert input: -x, then apply max pool, then invert output: -x
    return -max_pool2d(-input, kernel_size, stride=stride, padding=padding)

def find_k_same_class_pixels_excl_current(label_map, i, j, cls, min_count=5):
    """
    从 (i,j) 出发进行 BFS，寻找至少 min_count 个同类像素，**不包括 (i,j) 自身**
    label_map: torch.Tensor [H, W]
    i, j: 起始像素坐标
    cls: 当前像素类别
    min_count: 至少要找多少个同类像素
    返回: list of (x, y) 坐标（不含 (i,j)）
    """
    H, W = label_map.shape
    device = label_map.device
    visited = torch.zeros(H, W, dtype=torch.bool, device=device)
    queue = deque()
    result = []

    # 起始点标记为已访问，但不加入结果
    visited[i, j] = True
    queue.append((i, j))

    directions = [(-1,-1), (-1,0), (-1,1),
                  (0,-1),         (0,1),
                  (1,-1),  (1,0), (1,1)]

    while queue and len(result) < min_count:
        x, y = queue.popleft()

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and not visited[nx, ny]:
                visited[nx, ny] = True
                queue.append((nx, ny))
                if label_map[nx, ny] == cls:
                    result.append((nx, ny))

    return result[:min_count]  # 最多返回 min_count 个

def gather_pixels(tensor, coords):
    """
    从 tensor 中提取 coords 中指定的像素
    tensor: [C, H, W] 或 [H, W]
    coords: list of (x, y)
    返回: [K] 或 [C, K]
    """
    xs, ys = zip(*coords)
    xs = torch.tensor(xs, device=tensor.device)
    ys = torch.tensor(ys, device=tensor.device)

    if tensor.dim() == 2:
        return tensor[xs, ys]  # [K]
    elif tensor.dim() == 3:
        return tensor[:, xs, ys]  # [C, K]
    else:
        raise ValueError("Unsupported tensor dimension")

def find_k_nearest_pixels_var(label_map, preds, min_count=5):
    summit_var = torch.zeros_like(preds)
    valley_var = torch.zeros_like(preds)
    summit_var_t = torch.zeros_like(preds)
    valley_var_t = torch.zeros_like(preds)
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            indices_summit = find_k_same_class_pixels_excl_current(label_map, i, j, 1, min_count)
            pixels_summit = gather_pixels(preds, indices_summit)
            summit_var[i,j] = torch.var(pixels_summit)
            pixels_summit_t = torch.cat((pixels_summit, preds[i, j:j+1]), dim=0)
            summit_var_t[i,j] = torch.var(pixels_summit_t)

            indices_valley = find_k_same_class_pixels_excl_current(label_map, i, j, 0, min_count) 
            pixels_valley = gather_pixels(preds, indices_valley)
            valley_var[i,j] = torch.var(pixels_valley)
            pixels_valley_t = torch.cat((pixels_valley, preds[i, j:j+1]), dim=0)
            valley_var_t[i,j] = torch.var(pixels_valley_t)

    return summit_var, valley_var, summit_var_t, valley_var_t

def sum_val_filter(preds, turns=5):
    """
    Args:
        preds:(H, W), [0,1]
    Returns:
        result:(H, W)
    """
    preds = preds.unsqueeze(0).unsqueeze(0)
    summit = max_pool2d(preds, (3, 3), stride=1, padding=1)
    valley = min_pool2d(preds, (3, 3), stride=1, padding=1)
    result = torch.zeros_like(preds)

    gauss_kernel = gaussian_kernel(size=3, sigma=1.0, kernel_dim=2)

    # smoothy the summits and valleys
    summit = conv2d(summit, gauss_kernel, padding=1)
    valley_ = conv2d(valley, gauss_kernel, padding=1)
    valley = conv2d(valley_, gauss_kernel, padding=1)

    ## get original classification
    summit_wider = max_pool2d(summit, (3, 3), stride=1, padding=1)
    valley_wider = min_pool2d(valley, (3, 3), stride=1, padding=1)
    ## ignore the salience lower than 0.1
    condition1 = summit - valley > 0.1
    condition2 = summit + valley - 2 * preds < 0.
    coeficient = 1 - (10*E) ** (valley - summit)
    # result = ((summit - preds) > (preds - valley)) * ((summit - valley) > min_salience_thre)
    result = ((summit + valley) * coeficient + (summit_wider + valley_wider) * (1 - coeficient) - 2 * preds) < 0.
    result = condition1 * condition2
    # 处理不满足condition1的像素
    def cover_low_salience_pixels(condition_result, result):
        low_salience_pixels = ~condition_result
        kernel = torch.ones(3, 3).to(preds.device)
        kernel[1,1] = 0
        count = conv2d(result.float(), kernel.unsqueeze(0).unsqueeze(0), stride=1, padding=1)
        count = (count > 3) * low_salience_pixels
        result[low_salience_pixels] = count[low_salience_pixels]
        return result

    result = cover_low_salience_pixels(condition1, result)
    result = result.float()

    for i in range(turns):
        target_area = dilate_mask(result[0,0], 1)

        summit_var, valley_var, summit_var_t, valley_var_t = find_k_nearest_pixels_var(result[0,0], preds[0,0], 8)
        # 计算并比较方差变化后与原方差的比值
        def map_range(tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
            """
            将 tensor 线性映射到 [min_val, max_val]
            """
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Input must be a PyTorch tensor.")
                
            if tensor.numel() == 0:
                return tensor

            tensor_min = 0.
            tensor_max = 1.

            # 防止除以零：如果所有元素都相同，则返回全为 max_val 的 tensor
            if tensor_min == tensor_max:
                return torch.full_like(tensor, max_val)

            # 线性映射公式
            mapped_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
            mapped_tensor = mapped_tensor * (max_val - min_val) + min_val

            return mapped_tensor
       
        summit_var_time = map_range(summit_var_t, 1e-11, 1e11) / map_range(summit_var, 1e-11, 1e11)
        valley_var_time = map_range(valley_var_t, 1e-11, 1e11) / map_range(valley_var, 1e-11, 1e11)
 
        result_ = summit_var_time < valley_var_time # 将当前像素当成峰像素比例小于谷像素的，那么当前像素与峰像素较近，则将其分类为峰像素。
        result_ = result_.float() * target_area

        # 显示结果
        fig, axes = plt.subplots(4, 6)  # 可调整 figsize 控制整体大小
        ax = axes[0,0]
        ax.imshow(preds[0,0], cmap='gray', vmax=1., vmin=0.)

        ax = axes[0,1]
        ax.imshow(summit[0,0], cmap='gray', vmax=1., vmin=0.)

        ax = axes[0,2]
        ax.imshow(valley[0,0], cmap='gray', vmax=1., vmin=0.)

        ax = axes[0,3]
        ax.imshow(result[0,0], cmap='gray', vmax=1., vmin=0.)

        ax = axes[0,4]
        ax.imshow(target_area, cmap='gray', vmax=1., vmin=0.)

        ax = axes[1,0]
        ax.imshow(summit_var, cmap='gray')

        ax = axes[1,1]
        ax.imshow(valley_var, cmap='gray')

        ax = axes[1,2]
        ax.imshow(summit_var_t, cmap='gray')

        ax = axes[1,3]
        ax.imshow(valley_var_t, cmap='gray')

        ax = axes[1,4]
        ax.imshow(summit_var_time, cmap='gray')
        ax.set_title(f"min: {summit_var_time.min()}, max: {summit_var_time.max()}")

        ax = axes[1,5]
        ax.imshow(valley_var_time, cmap='gray')
        ax.set_title(f"min: {valley_var_time.min()}, max: {valley_var_time.max()}")

        ax = axes[2,0]
        ax.imshow(result_, cmap='gray', vmax=1., vmin=0.)
        
        plt.show()

        result = result_.unsqueeze(0).unsqueeze(0)

    return result.squeeze(0).squeeze(0)
