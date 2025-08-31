from PIL import Image
import numpy as np
import os
import os.path as osp


from utils.evaluation import SegmentationMetricTPFNFP


def evaluate_pseudo_mask(pesudo_mask_dir, mask_dir):
    names = os.listdir(mask_dir)

    metric = SegmentationMetricTPFNFP(nclass=1)

    for name in names:
        mask_path = osp.join(mask_dir, name)
        pesudo_mask_path = osp.join(pesudo_mask_dir, name)

        mask = Image.open(mask_path).convert('L')
        # mask = Image.open(mask_path).convert('L').resize((256, 256), Image.NEAREST)
        pesudo_mask = Image.open(pesudo_mask_path).convert('L') 

        mask, pesudo_mask = np.array(mask).astype(np.float32) / 255., np.array(pesudo_mask).astype(np.float32) / 255.
        # pesudo_mask 设置阈值使得其为二值图
        pesudo_mask = (pesudo_mask > 0.1).astype(np.float32)
        
        metric.update(mask, pesudo_mask)

    base_log = "pesudo_mask quality, mIoU: {:.4f}, prec: {:.4f}, recall: {:.4f}, F1: {:.4f} "
    iou , prec, recall, f1 = metric.get()
    print(base_log.format(iou , prec, recall, f1))