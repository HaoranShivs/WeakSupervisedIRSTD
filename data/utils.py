import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

import random
import numpy as np
import scipy.ndimage
import numpy as np
from PIL import Image


class Rotate_4D_Transform:
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, img):
        random_idx = torch.randint(0, 1000, (1,))
        
        angle = self.angles[random_idx%4]
        rotated_img = self.__rotate__(img, angle)
        return rotated_img
    
    def __rotate__(self, img, angle):
        C, _, _ = img.shape
        if angle == 90:
            img = torch.transpose(img, -1, -2)
            img = torch.flip(img, dims=(-1,)) 
        elif angle == 180:
            img = torch.flip(img, dims=(-1, -2))
        elif angle == 270:
            img = torch.transpose(img, -1, -2)
            img = torch.flip(img, dims=(-2,))
        return img
        

class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input.copy(), target.copy()


class RandomResize:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        # 随机选择一个目标大小
        target_size = random.randint(self.min_size, self.max_size)
        # 使用 Resize 调整图像大小
        resize_transform = transforms.Resize((target_size, target_size))
        return resize_transform(img)
    

class Augment_transform:
    def __init__(self, base_size=256, mode="train", cropResize_scale=(0.8, 1.0), affine_degrees=(-180, 180), affine_translate=(0.3, 0.3)):
        self.base_size = base_size
        self.mode = mode
        self.cropResize_scale = cropResize_scale
        self.cropratio = (1.0, 1.0)
        self.affine_degrees = affine_degrees
        self.affine_translate = affine_translate
        self.affine_scale = (1.0, 1.0)        # 缩放比例范围
        self.affine_shear = (0, 0)         # 剪切角度范围

    def __call__(self, img, mask):
        """
        Args:
            img: (C1,H,W), torch.tensor
            mask: (C2,H,W), torch.tensor
        Returns:
            img: (C1,H,W), torch.tensor
            mask: (C2,H,W), torch.tensor
        """

        if self.mode == "train":
            # 使用 RandomResizedCrop 的内部逻辑生成随机参数
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.cropResize_scale, ratio=self.cropratio)
            # 应用相同的随机参数到图像和 mask，分别使用不同的插值方式
            transformed_image = F.resized_crop(img, i, j, h, w, self.base_size, interpolation=Image.BILINEAR, antialias=True)  # 双线性插值
            transformed_mask = F.resized_crop(mask, i, j, h, w, self.base_size, interpolation=Image.BILINEAR, antialias=True)    # 最近邻插值

            # 使用 RandomAffine 的内部逻辑生成随机参数
            center = (self.base_size // 2, self.base_size // 2)
            angle, translations, scale_factor, shear_values = transforms.RandomAffine.get_params(
                degrees=self.affine_degrees,
                translate=self.affine_translate,
                scale_ranges=self.affine_scale,
                shears=self.affine_shear,
                img_size=(self.base_size, self.base_size))
            # 应用相同的随机参数到图像和 mask，分别使用不同的插值方式
            transformed_image = F.affine(
                transformed_image,
                angle=angle,
                translate=translations,
                scale=scale_factor,
                shear=shear_values,
                interpolation=Image.NEAREST,  # 双线性插值
                center=center
            )
            transformed_mask = F.affine(
                transformed_mask,
                angle=angle,
                translate=translations,
                scale=scale_factor,
                shear=shear_values,
                interpolation=Image.NEAREST,  # 最近邻插值
                center=center
            )
            # 随机生成是否进行水平翻转的概率
            p = torch.rand(1).item()  # 生成一个 [0, 1) 的随机数
            if p < 0.5:  # 如果概率小于 0.5，则进行水平翻转
                transformed_image = F.hflip(transformed_image)  # 水平翻转图像
                transformed_mask = F.hflip(transformed_mask)    # 水平翻转 mask
            else:
                transformed_image = transformed_image
                transformed_mask = transformed_mask
        elif self.mode == "test":
            transformed_image = F.resize(img, self.base_size, antialias=True)
            transformed_mask = F.resize(mask, self.base_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        else:
            raise ValueError("mode should be train or test") 

        return transformed_image, transformed_mask
    

# def mix_target(img, mask):
#     """
#     Mix the target with image and mask.
#     Target is from perferct generated pesudo label, with no dissociated pixels.
#     Mixing includs the following steps:
#     1. Find the proper position for the target where the img is complex. Furthermore, complex area means there are many edges.
#     2. Mix the target with image in proper way which means the border of the target and background is smooth.
#     3. Make mask according to the new and original target.
#     """
#     # edge_path = osp.join(self.data_dir, "canny_edge")
#     # name = self.names[i]
#     # edge = cv2.imread(osp.join(edge_path, name), 0)
#     # edge = torch.from_numpy(edge).type(torch.float32)
#     # edge = self.augment_test(edge.unsqueeze(0)) / 255.0

#     target_path = osp.join(self.data_dir, "perfect_target")
#     target_names = os.listdir(target_path)

#     # compute how many targets are needed
#     seed = random.random() * 4
#     target_num = 0
#     if seed > 1.:
#         target_num = int(seed)
#     else:
#         target_num = 1 if random.random() <= seed else 0
        
#     # target trasform
#     while target_num > 0:
#         target_name = random.choice(target_names)
#         target = cv2.imread(osp.join(target_path, target_name), 0)
#         target = torch.from_numpy(target).type(torch.float32) / 255.0

#         target = RandomResize(32, 64)(target.unsqueeze(0))
#         target = self.target_trans(target)
#         _, iH, iW = img.shape
#         _, tH, tW = target.shape
#         # target_blured = self.gaussian_blur(target.unsqueeze(0))
#         # # plt.figure(figsize=(12, 12))
#         # # plt.imshow(target_blured[0], cmap='gray')
#         # # plt.show()
#         # # a = input()

#         # Step 1, find an area where the complexicity is mid-level.
#         # h_idx, w_idx = self.__random_position(edge, mask)
#         h_idx, w_idx = random.randint(0, iH-1), random.randint(0, iW-1)
#         ## coordinate of the target 
#         x1, x2 = max(w_idx - tW // 2, 0), min(w_idx + (tW - tW // 2), iW-1)
#         y1, y2 = max(h_idx - tH // 2, 0), min(h_idx + (tH - tH // 2), iH-1)
#         ## part of the target
#         tx1, tx2 = tW // 2 - (w_idx - x1), x2 - w_idx + tW // 2
#         ty1, ty2 = tH // 2 - (h_idx - y1), y2 - h_idx + tH // 2
#         # mix the target with image(simple way)
#         img_part = img[0, y1:y2, x1:x2].clone()
#         target = target[0, ty1:ty2, tx1:tx2]
#         if target.max() <= 0.1:
#             return img, mask
#         target_blured = self.gaussian_blur(target.unsqueeze(0))[0]
#         img_accor_part = img_part * (target_blured > 0.)
#         # plt.figure(figsize=(20, 5))
#         # plt.subplot(141), plt.imshow(target, cmap='gray')
#         # plt.subplot(142), plt.imshow(target_blured, cmap='gray')
#         # plt.subplot(143), plt.imshow(img_accor_part, cmap='gray')
#         # plt.subplot(144), plt.imshow(img[0, y1:y2, x1:x2], cmap='gray')
#         # plt.show()
#         # a = input()
#         ## contrast modification
#         target_mean = target[target > 0.1].mean()
#         img_accor_part_mean = img_accor_part[target > 0.1].mean()
#         img_accor_part_mean = img_accor_part_mean if img_accor_part_mean > 0.1 else 0.1
#         enhanced_ratio = random.uniform(1.6, 2.)    # super parameter
#         adaptive_intensity_ratio = (enhanced_ratio - 1) * img_accor_part_mean / target_mean
#         target_enbeded = target * adaptive_intensity_ratio + img_accor_part_mean * (target > 0.) 
#         ## mixing
#         target_area = self.gaussian_blur(target_enbeded.unsqueeze(0))[0]
#         traget_blured = self.gaussian_blur((target_enbeded + (target_enbeded <= 0.1) * img_part).unsqueeze(0))[0] * (target_area > 0.001)

#         # anti_blured_target = self.gaussian_blur((target_blured > 0.1).type(torch.float32).unsqueeze(0))[0]
#         # img[0, y1:y2, x1:x2] = (target_enbeded * (target > 0.1) + mixing_blured * (target <= 0.1)) * anti_blured_target + img_part * (1-anti_blured_target)
#         # img[0, y1:y2, x1:x2] = (target_blured * (target_edge_blured > 0.2) + target_edge_blured * (target_edge_blured <= 0.2))* anti_blured_target + img_part * (1-anti_blured_target)
#         # img[0, y1:y2, x1:x2] = (target_blured * (target > 0.1) + img_accor_part_mean * (target_blured>0.) * (target <= 0.1)) * anti_blured_target + img_part * (1-anti_blured_target)
#         img[0, y1:y2, x1:x2] = traget_blured + img_part * (target_area <= 0.001)
#         # img[0, y1:y2, x1:x2] = target_enbeded * anti_blured_target + img_part * (1-anti_blured_target)
#         img[0, y1:y2, x1:x2] = torch.clamp_max(img[0, y1:y2, x1:x2], 1.0)

#         # mix the target with mask
#         mask[0, y1:y2, x1:x2] = mask[0, y1:y2, x1:x2] + (target_area > 0.1).type(torch.float32)

#         target_num -= 1

#         # plt.figure(figsize=(20, 5))
#         # plt.subplot(141), plt.imshow(target_blured, cmap='gray', vmin=0., vmax=1.)
#         # plt.subplot(142), plt.imshow(target_area, cmap='gray', vmin=0., vmax=1.)
#         # plt.subplot(143), plt.imshow(img[0, y1:y2, x1:x2], cmap='gray', vmin=0., vmax=1.)
#         # plt.subplot(144), plt.imshow(mask[0, y1:y2, x1:x2], cmap='gray', vmin=0., vmax=1.)
#         # print(adaptive_intensity_ratio)
#         # plt.show()
#         # a = input()
#     return img, mask

# def random_position(self, edge, mask):
#     """
#     Randomly choose a position for the target where there are some edges.
#     """
#     H, W = edge.shape
#     edge_level = torch.nn.functional.avg_pool2d(edge.unsqueeze(0).unsqueeze(0), 2, stride=2)  # (1,1,128,128)
#     edge_level = torch.nn.functional.avg_pool2d(edge_level, 2, stride=2)    # (1,1,64,64)
#     edge_level = torch.nn.functional.avg_pool2d(edge_level, 2, stride=2)    # (1,1,32,32)

#     edge_mean = edge_level.mean()
#     edge_mid = (edge_level.max() + edge_level.min()) / 2
#     condition = (edge_level > edge_mean) * (edge_level < edge_mid)  # ??? is it proper?
#     _, _, H_idx, W_idx = torch.where(condition)
#     # to makesure we can get target position even there is no proper edge area.
#     make_sure = 10 
#     if H_idx.shape[0] < make_sure:
#         H_idx = torch.concatenate([H_idx, torch.randint(1, H//8, (make_sure - H_idx.shape[0],))], dim=0)
#         W_idx = torch.concatenate([W_idx, torch.randint(1, W//8, (make_sure - W_idx.shape[0],))], dim=0)

#     random_idx = torch.randint(0, H_idx.shape[0], (1,))
#     rh_idx, rw_idx = int(H_idx[random_idx].item()), int(W_idx[random_idx].item())
#     rh_idx, rw_idx = rh_idx * 8 + 4, rw_idx * 8 + 4
#     # notice: new position is supposed to not overlap with original target, because that will decrease the multi-targets' effect
#     while mask[0, rh_idx, rw_idx].max() > 0.1:
#         random_idx = torch.randint(0, H_idx.shape[0], (1,))
#         rh_idx, rw_idx = int(H_idx[random_idx].item()), int(W_idx[random_idx].item())
#         rh_idx, rw_idx = rh_idx * 8 + 4, rw_idx * 8 + 4

#     return rh_idx, rw_idx

def mask2point(mask, img, offset=3):
    # 将mask和img转换为numpy数组
    base_size = mask.shape[-1]
    mask_array = np.array(mask[0].cpu())  # 假设mask是tensor类型
    img_array = np.array(img[0].cpu())  # H x W x C 格式

    # 使用连通组件分析找到所有独立的目标区域
    labels, num_features = scipy.ndimage.label(mask_array > 0.9)

    pts_label = torch.zeros_like(mask, dtype=torch.float32)

    for label_id in range(1, num_features + 1):
        target_mask = labels == label_id
        coords = np.argwhere(target_mask)

        if len(coords) == 0:
            continue

        # 获取目标区域内的图像像素值及其坐标
        masked_img_vals = img_array[target_mask]
        # print(masked_img_vals.shape)
        masked_coords = coords

        # 计算每个像素的亮度（例如灰度值）
        brightness = masked_img_vals.flatten()

        # 找出最亮部分的像素
        threshold = np.percentile(brightness, 20)
        bright_coords = masked_coords[brightness >= threshold]

        # 如果没有亮点，跳过
        if len(bright_coords) == 0:
            continue

        # 计算原质心
        centroid = np.mean(coords, axis=0).astype(np.int64)
        if target_mask[centroid[0], centroid[1]] == 0:
            corrd_diff = coords - centroid
            min_dist_idx = np.argmin(np.sum(corrd_diff**2, axis=1))
            centroid = coords[min_dist_idx]

        # 找到离质心最近的亮点
        dists = np.linalg.norm(bright_coords - centroid, axis=1)
        nearest_point = bright_coords[np.argmin(dists)]
        if nearest_point.ndim > 1:
            nearest_point = nearest_point[0]

        point_y_x = nearest_point.copy()

        # 开始偏移尝试（在最亮的10%像素中找最近的点）
        attempt_count = 0
        while attempt_count < 10 and offset > 0:
            # 随机偏移
            offset_y_x = np.random.uniform(-offset, offset, (2,)).astype(np.int64)
            new_point = point_y_x + offset_y_x
            new_point = np.clip(new_point, 0, np.array([base_size-1, base_size-1]))

            # 找到离new_point最近的亮点
            dists = np.linalg.norm(bright_coords - new_point, axis=1)
            nearest_point_candidate = bright_coords[np.argmin(dists)]

            if np.array_equal(nearest_point_candidate, point_y_x):
                offset -= 1
                attempt_count += 1
                continue
            else:
                point_y_x = nearest_point_candidate
                break

        # 设置最终的点标签
        pts_label[0, point_y_x[0], point_y_x[1]] = 1.

    return pts_label