import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLoULoss(nn.Module):
    def __init__(self):
        super(SoftLoULoss, self).__init__()

    def forward(self, pred, target):
        smooth = 0.1
        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)
        return loss


class SoftIoUL1NromLoss(nn.Module):
    def __init__(self, lambda_iou=0.8, lambda_l1=0.2):
        super(SoftIoUL1NromLoss, self).__init__()
        self.softiou = SoftLoULoss()
        self.lambda_iou = lambda_iou
        self.lambda_l1 = lambda_l1

    def forward(self, pred, target):
        iouloss = self.softiou(pred, target)

        batch_size, C, height, width = pred.size()
        pred = (pred > 0).float()
        l1loss = torch.sum(pred) / (batch_size * C * height * width)

        loss = self.lambda_iou * iouloss + self.lambda_l1 * l1loss
        return loss


class Heatmap_SoftIoU(nn.Module):
    def __init__(self, cfg):
        super(Heatmap_SoftIoU, self).__init__()
        self.softIouLoss = SoftLoULoss()
        self.cfg = cfg

    def forward(self, preds, label):
        labels = self.downsampleImg(label)
        loss1 = self.softIouLoss(preds[0], labels[2])
        loss2 = self.softIouLoss(preds[1], labels[1])
        loss3 = self.softIouLoss(preds[2], labels[0])
        loss = (loss1 + loss2 + loss3) / 3 
        return loss
    
    def downsampleImg(self, img):
        downsampler = nn.MaxPool2d(2, 2)

        downsampled_imgs = []
        downsampled_img = img

        for i in range(len(self.cfg["multiscalefeature_outchannel"]), 0, -1):
            downsampled_img = downsampler(downsampled_img)
            if i <= 3:
                downsampled_imgs.append(downsampled_img)
        
        return downsampled_imgs


class Heatmap_MSE(nn.Module):
    def __init__(self, cfg):
        super(Heatmap_MSE, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.cfg = cfg

    def forward(self, preds, label):
        # make the vague caused by augment transform clear
        label = (label > self.cfg["label_vague_threshold"]).type(torch.float32)
        labels = self.downsampleImg(label)
        loss1 = self.mse_loss(preds[0], labels[2])
        loss2 = self.mse_loss(preds[1], labels[1])
        loss3 = self.mse_loss(preds[2], labels[0])
        loss = (loss1 + loss2 + loss3) / 3 
        return loss
    
    def downsampleImg(self, img):
        downsampler = nn.AvgPool2d(2, 2)

        downsampled_imgs = []
        downsampled_img = img

        for i in range(len(self.cfg["multiscalefeature_outchannel"]), 0, -1):
            downsampled_img = downsampler(downsampled_img)
            if i <= 3:
                downsampled_imgs.append(downsampled_img)
        
        return downsampled_imgs


class ImageRecoverLoss(nn.Module):
    def __init__(self, loss_weight_per_scale):
        super(ImageRecoverLoss, self).__init__()
        self.weight = loss_weight_per_scale
        self.avg_pool = nn.AvgPool2d(2, 2, 0)
        

    def forward(self, generP, oriP):

        z = torch.cat((generP, oriP), 1)

        loss = 0
        for i in self.weight:
            z = self.avg_pool(z)
            loss_ = torch.abs(z[:,0,...] - z[:,1,...]).sum() / (z.shape[0] * z.shape[2] * z.shape[3])
            loss += loss_ * i
        
        return loss
    

class Detail_loss(nn.Module):
    def __init__(self, dilation_kernel_size=5):
        super(Detail_loss, self).__init__()
        self.dilation_kernel_size = dilation_kernel_size

    def forward(self, preds, labels, images):
        # 对标签进行膨胀
        dilated_labels = self.dilate_labels(labels, self.dilation_kernel_size) if self.dilation_kernel_size > 0 else labels
        # # 将膨胀后的标签与原图相乘，得到抠图
        # cropped_images = self.min_max_normalize(images * dilated_labels)
        # cropped_images = (cropped_images * 0.7 + 0.3) * dilated_labels
        images = images * dilated_labels
        cropped_preds = preds * dilated_labels
        B, _, _, _ = labels.shape
        cropped_images = torch.zeros_like(images)
        for i in range(B):
            img = images[i, 0][dilated_labels[i,0]>0]
            thre1, thre2 = self.multi_otsu_threshold(img)
            cropped_images[i, 0][images[i, 0] >= thre2] = 1.
            cropped_images[i, 0][(images[i, 0] >= thre1) & (images[i, 0] < thre2)] = 0.5
        # 计算损失
        loss = self.compute_weighted_mse(cropped_images, cropped_preds, dilated_labels)
        
        return loss
    
    def dilate_labels(self, labels, dilation_kernel_size=5):
        # 创建一个膨胀内核
        kernel = torch.ones(1, 1, dilation_kernel_size, dilation_kernel_size, device=labels.device, requires_grad=False)
        # 应用膨胀操作
        dilated_labels = F.conv2d(labels, kernel, padding=dilation_kernel_size//2)
        dilated_labels = (dilated_labels > 0).float()  # 转换为二值标签
        return dilated_labels
    
    def compute_weighted_mse(self, cropped_images, cropped_preds, dilated_labels):
        # 计算平方差
        squared_diff = (cropped_images - cropped_preds).pow(2)
        # 按高度和宽度求和
        sum_squared_diff = torch.sum(squared_diff, dim=(2, 3))
        # 计算膨胀标签的总和
        sum_dilated_labels = torch.sum(dilated_labels, dim=(2, 3)) + 1e-8
        # 检查哪些样本有目标物
        valid_samples = sum_dilated_labels > 1e-8
        # 只对有目标物的样本计算损失
        if valid_samples.any():
            normalized_loss = sum_squared_diff[valid_samples] / sum_dilated_labels[valid_samples]
            mean_loss = torch.mean(normalized_loss)
        else:
            # 如果没有有效样本，返回0损失
            mean_loss = torch.tensor(0.0, device=cropped_images.device)
        return mean_loss
    
    def min_max_normalize(self, images):
        # 计算每张图像的最小值和最大值
        min_vals = images.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        max_vals = images.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        # 避免除以零的情况
        range_vals = max_vals - min_vals + 1e-8
        # 归一化
        normalized_images = (images - min_vals) / range_vals
        return normalized_images
    
    def multi_otsu_threshold(self, image): 
        # 将图像转换为PyTorch张量
        image_tensor = image*255
        
        # 计算直方图
        hist = torch.histc(image_tensor, bins=256, min=0, max=255)
        hist = hist / hist.sum()  # 归一化直方图
        
        # 计算累积概率和累积均值
        cum_hist = torch.cumsum(hist, dim=0)
        cum_mean = torch.cumsum(hist * torch.arange(256, dtype=torch.float32), dim=0)
        
        # 总均值
        total_mean = cum_mean[-1]
        
        # 创建所有可能的阈值组合
        t1 = torch.arange(1, 255, dtype=torch.int64)
        t2 = torch.arange(1, 255, dtype=torch.int64)
        
        # 生成网格
        t1_grid, t2_grid = torch.meshgrid(t1, t2, indexing='ij')
        
        # 计算各部分的权重
        w0 = cum_hist[t1_grid - 1]
        w1 = cum_hist[t2_grid - 1] - cum_hist[t1_grid - 1]
        w2 = 1 - cum_hist[t2_grid - 1]
        
        # 排除无效的权重组合
        valid_mask = (w0 > 0) & (w1 > 0) & (w2 > 0)
        
        # 计算各部分的均值
        smooth = 1e-8
        mean0 = cum_mean[t1_grid - 1] / (w0 + smooth)
        mean1 = (cum_mean[t2_grid - 1] - cum_mean[t1_grid - 1]) / (w1 + smooth)
        mean2 = (total_mean - cum_mean[t2_grid - 1]) / (w2 + smooth)
        
        # 计算类间方差
        between_var = w0 * (mean0 - total_mean) ** 2 + w1 * (mean1 - total_mean) ** 2 + w2 * (mean2 - total_mean) ** 2
        
        # 只考虑有效的权重组合
        between_var[~valid_mask] = 0
        
        # 找到最大类间方差对应的阈值
        max_var_idx = torch.argmax(between_var)
        optimal_t1, optimal_t2 = t1_grid.flatten()[max_var_idx], t2_grid.flatten()[max_var_idx]
    
        return optimal_t1.item()/255, optimal_t2.item()/255