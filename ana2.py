import numpy as np
from PIL import Image
from collections import Counter
import math

def get_candidate_periods_1d(binary_line, min_period=2, max_period=None):
    """
    从一维二值序列中找出可能的周期（块大小）。
    使用差分 + 距离直方图方法。
    """
    if max_period is None:
        max_period = len(binary_line) // 2

    # 找变化点
    changes = np.where(np.diff(binary_line) != 0)[0] + 1
    if len(changes) < 2:
        return []

    distances = np.diff(changes)
    distances = distances[(distances >= min_period) & (distances <= max_period)]

    if len(distances) == 0:
        return []

    # 统计距离频率
    counter = Counter(distances)
    total = sum(counter.values())
    candidates = []
    for dist, count in counter.most_common():
        if count / total >= 0.3:  # 至少 30% 的距离支持这个周期
            candidates.append(dist)
    return candidates

def infer_block_size_robust(mask_2d, max_candidates=5):
    """
    从 2D 二值 mask 中鲁棒地推断块大小 (block_h, block_w)
    """
    h, w = mask_2d.shape

    # 确保是 0/1
    mask = (mask_2d > 0).astype(np.uint8)

    # 如果全黑或全白，无法推断
    if mask.sum() == 0 or mask.sum() == h * w:
        print("警告：图像全黑或全白，无法推断分辨率。默认返回 32x32。")
        return h // 32 if h >= 32 else 1, w // 32 if w >= 32 else 1

    # 收集水平方向（宽度）的候选块大小
    hor_candidates = []
    rows_to_sample = min(10, h)
    step = max(1, h // rows_to_sample)
    for i in range(0, h, step):
        cands = get_candidate_periods_1d(mask[i, :], min_period=2, max_period=w//4)
        hor_candidates.extend(cands)

    # 收集垂直方向（高度）的候选块大小
    ver_candidates = []
    cols_to_sample = min(10, w)
    step = max(1, w // cols_to_sample)
    for j in range(0, w, step):
        cands = get_candidate_periods_1d(mask[:, j], min_period=2, max_period=h//4)
        ver_candidates.extend(cands)

    # 如果没有候选，回退到简单策略
    if not hor_candidates:
        hor_candidates = [w // 32] if w >= 32 else [1]
    if not ver_candidates:
        ver_candidates = [h // 32] if h >= 32 else [1]

    # 取众数作为最终块大小
    block_w = Counter(hor_candidates).most_common(1)[0][0]
    block_h = Counter(ver_candidates).most_common(1)[0][0]

    # 验证：块大小必须能大致整除图像尺寸
    # 找最接近的因数
    def closest_divisor(size, approx_block):
        if approx_block <= 1:
            return 1
        best = approx_block
        min_diff = abs(size % approx_block)
        for b in range(max(1, approx_block - 5), approx_block + 6):
            if b <= 0:
                continue
            diff = size % b
            if diff == 0:
                return b
            if diff < min_diff:
                min_diff = diff
                best = b
        return best

    block_w = closest_divisor(w, block_w)
    block_h = closest_divisor(h, block_h)

    return block_h, block_w

def extract_lowres_mask_robust(screenshot_path, fallback_size=(32, 32)):
    """
    鲁棒地从 mask 截图中提取原始低分辨率 mask。
    自动推断分辨率，失败时回退到 fallback_size。
    返回: (H, W) 的 0/1 numpy 数组
    """
    # 读取并二值化
    img = Image.open(screenshot_path).convert('L')
    arr = np.array(img)
    mask = (arr > 127).astype(np.uint8)  # 转为 0/1
    h, w = mask.shape

    try:
        block_h, block_w = infer_block_size_robust(mask)
        orig_h = h // block_h
        orig_w = w // block_w

        # 安全检查
        if orig_h < 1 or orig_w < 1:
            raise ValueError("推断分辨率无效")

        print(f"✅ 推断成功: 原始分辨率 = {orig_h}x{orig_w}, 块大小 = {block_h}x{block_w}")
    except Exception as e:
        print(f"⚠️ 自动推断失败 ({e})，使用回退分辨率: {fallback_size}")
        orig_h, orig_w = fallback_size

    # 使用 NEAREST 重采样（对二值图最安全）
    pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
    lowres_pil = pil_mask.resize((orig_w, orig_h), Image.Resampling.NEAREST)
    lowres = (np.array(lowres_pil) > 127).astype(np.uint8)

    return lowres

# ================== 使用示例 ==================
if __name__ == "__main__":
    screenshot_file = "picts/page010_img005.png"  # ← 替换为你的文件路径

    lowres_mask = extract_lowres_mask_robust(
        screenshot_file,
        fallback_size=(32, 32)  # 如果推断失败，用这个
    )

    print("输出形状:", lowres_mask.shape)
    print("像素值:", np.unique(lowres_mask))

    # 可视化（放大 10 倍）
    vis = Image.fromarray((lowres_mask * 255).astype(np.uint8))
    vis = vis.resize((lowres_mask.shape[1]*10, lowres_mask.shape[0]*10), Image.Resampling.NEAREST)
    vis.show()