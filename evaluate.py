import os
import os.path as osp
import time
import datetime
from argparse import ArgumentParser
from PIL import Image

import yaml
import numpy as np
import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

# from models import get_model
from dataprocess.sirst import NUDTDataset, IRSTD1kDataset
# from net.basenet import BaseNet1, BaseNet2, LargeBaseNet, LargeBaseNet2, BaseNet3, GaussNet, GaussNet3, GaussNet4, SigmoidNet
# from net.twotasknet import LocalSegment, TwoTaskNetWithLoss
# from net.attentionnet import attenMultiplyUNet_withloss
from net.basenet import BaseNet4, BaseNetWithLoss
from utils.loss import SoftLoULoss
from utils.lr_scheduler import *
from utils.evaluation import SegmentationMetricTPFNFP, my_PD_FA
from utils.logger import setup_logger
from net.dnanet import DNANet_withloss, Res_CBAM_block




def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Evaluation of BaseNet')

    parser.add_argument('--gpu', type=str, default='0', help='GPU number')
    parser.add_argument("--seed", type=int, default=1, help="seed")
    #
    # Dataset parameters
    #
    parser.add_argument('--base-size', type=int, default=256, help='base size of images')
    parser.add_argument('--crop-size', type=int, default=256, help='crop size of images')
    parser.add_argument('--dataset', type=str, default='nudt', help='choose datasets')

    #
    # Net path
    #
    parser.add_argument('--model-path', type=str, default='result/20240620T10-46-34_basenet_16-32-48-64-128_nudt',
                        help='net name: fcn')
    
    args = parser.parse_args()

    # seed
    if args.seed != 0:
        set_seeds(args.seed)

    return args

def set_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Evaluate(object):
    def __init__(self, args):
        self.args = args

        ## cfg file
        cfg_path = osp.join(args.model_path, 'cfg.yaml')
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        # dataset
        if args.dataset == 'nudt':
            valset = NUDTDataset(base_dir=r'W:/DataSets/ISTD/NUDT-SIRST', mode='test', base_size=args.base_size, cfg=self.cfg)
        # elif args.dataset == 'sirstaug':
        #     trainset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
        #                                mode='train', base_size=args.base_size)  # base_dir=r'E:\ztf\datasets\sirst_aug'
        #     valset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
        #                              mode='test', base_size=args.base_size)  # base_dir=r'E:\ztf\datasets\sirst_aug'
        elif args.dataset == 'irstd1k':
            valset = IRSTD1kDataset(base_dir=r'W:/DataSets/ISTD/IRSTD-1k', mode='test', base_size=args.base_size, cfg=self.cfg) # base_dir=r'E:\ztf\datasets\IRSTD-1k'
        else:
            raise NotImplementedError

        self.val_data_loader = Data.DataLoader(valset, batch_size=32, shuffle=False, drop_last=False)

        ## GPU
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

        ## model
        # net = BaseNet4(1, self.cfg)
        # loss_fn = SoftLoULoss()
        # self.net = BaseNetWithLoss(self.cfg, net, loss_fn)
        self.net = DNANet_withloss(1, 
                    input_channels=3, 
                    block=Res_CBAM_block,
                    num_blocks=[2, 2, 2, 2],
                    nb_filter=[16, 32, 64, 128, 256],
                    deep_supervision=True,
                    grad_loss=True)

        ## load_model
        model_path = osp.join(args.model_path, 'best.pkl')
        self.net.load_state_dict(torch.load(model_path))
        self.net = self.net.to(self.device)
        self.net.eval()

        # self.softiou = SoftLoULoss()

        ## evaluation metrics
        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.best_miou = 0
        self.best_fmeasure = 0
        self.eval_loss = 0  # tmp values
        self.miou = 0
        self.fmeasure = 0
        self.eval_my_PD_FA = my_PD_FA()

        ## visualization
        self.best_ori_pict = None
        self.best_seg_pict = None
        self.best_seg_label = None
    

    def validation(self):
        self.metric.reset()
        # self.eval_my_PD_FA.reset()
        base_log = "Data: {:s}, mIoU: {:.4f}/{:.4f}, prec: {:.4f}/{:.4f}, recall: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f} "
        # base_log = "Data: {:s}, mIoU: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f}, Pd:{:.4f}, Fa:{:.8f} "
        for i, (data, labels) in enumerate(self.val_data_loader):
            data = data.repeat(1, 3, 1, 1)  # for DNANet
            data = data.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                # noise = torch.zeros((data.shape[0], 1, 32, 32), device=data.device)
                # pred, _, _, _, _ = self.net.net(data.to(self.device))
                pred,_ = self.net(data.to(self.device), (labels > 0.1).float(), labels)
                pred = pred[-1]
            out_T = pred.cpu() 

            # loss_softiou = self.softiou(out_T, labels)
            # loss_mse = self.mse(out_D, data)
            # gamma = torch.Tensor([0.1]).to(self.device)
            # loss_all = loss_softiou + torch.mul(gamma, loss_mse)
            
            labels = (labels > self.cfg["label_vague_threshold"]).type(torch.float32).cpu()
            self.metric.update(labels, out_T)
            for j in range(labels.shape[0]):
                self.eval_my_PD_FA.update(out_T[j,0], labels[j,0])

        miou, prec, recall, fmeasure = self.metric.get()
        pd, fa = self.eval_my_PD_FA.get()
        print(miou, fmeasure, pd, fa)

    def visualize(self):
        oripict = (np.array(self.best_ori_pict) * 255).astype(np.uint8)
        segpict = (np.array(self.best_seg_pict) * 255).astype(np.uint8)
        label = (np.array(self.best_seg_label) * 255).astype(np.uint8)
        
        # 验证是否只有0和255
        _, H, W = label[0].shape
        for i in range(32):
            mask_ = label[i]
            max_index = np.argmax(mask_)
            print(max_index)
            row_idx, col_dix = int(max_index/W), int(max_index - int(max_index/W) * W)
            # if(mask_[0, row_idx, col_dix] < 255 and mask_[0, row_idx, col_dix] != 0):
            print(mask_[0,row_idx-7:row_idx+7, col_dix-7: col_dix+5])

        n = segpict.shape[0]  # 图片对的数量
        shape = label.shape[-2:]

        # 创建一个n行2列的子图网格
        fig, axes = plt.subplots(n, 3, figsize=(6, 3*n))  # 调整figsize以适应你的需要

        for i in range(n):
            # 在当前行的第一列显示第1张图片
            axes[i, 0].imshow(oripict[i].reshape(*shape), cmap='gray', vmin=0, vmax=255)
            axes[i, 0].axis('off')  # 关闭坐标轴显示
            
            # 在当前行的第一列显示第2张图片
            axes[i, 1].imshow(segpict[i].reshape(*shape), cmap='gray', vmin=0, vmax=255)
            axes[i, 1].axis('off')  # 关闭坐标轴显示
            
            # 在当前行的第二列显示第3张图片
            axes[i, 2].imshow(label[i].reshape(*shape), cmap='gray', vmin=0, vmax=255)
            axes[i, 2].axis('off')  # 同样关闭坐标轴显示

        # 显示图表
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        # 保存图表、
        save_path= osp.join(self.args.model_path, args.dataset + '_result.png')
        plt.savefig(save_path)

    def visualize2(self):
        oripict = (np.array(self.best_ori_pict) * 255).astype(np.uint8)
        segpict = (np.array(self.best_seg_pict) * 255).astype(np.uint8)
        label = (np.array(self.best_seg_label) * 255).astype(np.uint8)

        n = segpict.shape[0]  # 图片对的数量
        shape = label.shape[-2:]

        # 每英寸点数 (dpi)
        dpi = 300
        
        # 每个子图的 figsize
        figsize_per_image = (shape[1] / dpi, shape[0] / dpi)
        
        # 整个 figure 的 figsize
        figsize = (figsize_per_image[0] * 3, figsize_per_image[1] * n)  # 三列子图，n行
        
        # 创建一个n行3列的子图网格
        fig, axes = plt.subplots(n, 3, figsize=figsize, )

        for i in range(n):
            # 在当前行的第一列显示第1张图片
            axes[i, 0].imshow(oripict[i].reshape(*shape), cmap='gray', vmin=0, vmax=255)
            axes[i, 0].axis('off')  # 关闭坐标轴显示
            
            # 在当前行的第二列显示第2张图片
            axes[i, 1].imshow(segpict[i].reshape(*shape), cmap='gray', vmin=0, vmax=255)
            axes[i, 1].axis('off')  # 关闭坐标轴显示
            
            # 在当前行的第三列显示第3张图片
            axes[i, 2].imshow(label[i].reshape(*shape), cmap='gray', vmin=0, vmax=255)
            axes[i, 2].axis('off')  # 同样关闭坐标轴显示

        # 显示图表
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        # 保存图表
        save_path = osp.join(self.args.model_path, self.args.dataset + '_result.png')
        plt.savefig(save_path, bbox_inches='tight')

    def visualize3(self, gap_width=5):
        # 定义参数
        num_images = self.best_seg_label.shape[0]  # 图片对的数量
        shape = self.best_seg_label.shape[-2:]

        oripict = (np.array(self.best_ori_pict) * 255).astype(np.uint8)
        segpict = (np.array(self.best_seg_pict) * 255).astype(np.uint8)
        label = (np.array(self.best_seg_label) * 255).astype(np.uint8)
        
        # 创建空白图像用于间隙
        gap_horizontal = np.ones((shape[0], gap_width), dtype=np.uint8) * 255  # 创建一个全白的间隙
        gap_vertical = np.ones((gap_width* 2, shape[1]*3 + 2 * gap_width), dtype=np.uint8) * 255  # 创建一个全白的垂直间隙


        # 拼接单个实例
        def concatenate_images(image1, image2, image3):
            # 水平拼接图像
            combined = np.concatenate((image1, gap_horizontal, image2, gap_horizontal, image3), axis=1)
            return combined

        # 拼接所有实例
        all_combined = []
        for i in range(num_images):
            # 获取单个实例的图像
            img_pict = oripict[i, 0]
            img_predict_seg = segpict[i, 0]
            img_label_seg = label[i, 0]
            
            # 拼接单个实例
            combined_instance = concatenate_images(img_pict, img_predict_seg, img_label_seg)
            
            # 添加到列表
            all_combined.append(combined_instance)

        # 垂直拼接所有实例
        final_image = all_combined[0]
        for i in range(1, num_images):
            # 添加垂直间隙
            final_image = np.concatenate((final_image, gap_vertical, all_combined[i]), axis=0)

        # 添加外边框
        outer_gap_vertical = np.ones((gap_width*2, final_image.shape[1]), dtype=np.uint8) * 255
        outer_gap_horizontal = np.ones((final_image.shape[0]+gap_width*4, gap_width*2), dtype=np.uint8) * 255

        # 上下添加空白
        final_image_with_top_bottom_gap = np.concatenate((outer_gap_vertical, final_image, outer_gap_vertical), axis=0)

        # 左右添加空白
        final_image_with_outer_gaps = np.concatenate((outer_gap_horizontal, final_image_with_top_bottom_gap, outer_gap_horizontal), axis=1)

        # 转换为PIL图像
        pil_image = Image.fromarray(final_image_with_outer_gaps)

        # 保存图像
        save_path= osp.join(self.args.model_path, args.dataset + '_result.png')
        pil_image.save(save_path)

if __name__ == '__main__':
    args = parse_args()

    trainer = Evaluate(args)
    trainer.validation()
    # trainer.visualize3()