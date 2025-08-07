import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# __all__ = ['SegmentationMetricTPFNFP']

def get_miou_prec_recall_fscore(total_tp, total_fp, total_fn):
    miou = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp + total_fn)
    prec = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp)
    recall = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fn)
    fscore = 2.0 * prec * recall / (np.spacing(1) + prec + recall)

    return miou, prec, recall, fscore


class SegmentationMetricTPFNFP(object):
    """
    Computes pixAcc and mIoU metric scroes
    """
    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            tp, fp, fn = batch_tp_fp_fn(pred, label, self.nclass)
            # print(self.total_tp, self.total_fp, self.total_fn)
            with self.lock:
                self.total_tp += tp
                self.total_fp += fp
                self.total_fn += fn
            return

        if isinstance(preds, torch.Tensor):
            preds = (preds.detach().numpy() > 0.5).astype('int64')  # P
            labels = labels.numpy().astype('int64')  # T
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        #elif preds.dtype == numpy.uint8:
        elif isinstance(preds, np.ndarray):
            preds = ((preds / np.max(preds)) > 0.5).astype('int64')  # P
            labels = (labels / np.max(labels)).astype('int64')  # T
            evaluate_worker(self, labels, preds)
        else:
            raise NotImplemented

    def get_all(self):
        return self.total_tp, self.total_fp, self.total_fn

    def get(self):
        return get_miou_prec_recall_fscore(self.total_tp, self.total_fp, self.total_fn)

    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        return

def batch_tp_fp_fn(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """

    mini = 1
    maxi = nclass
    nbins = nclass

    # predict = (output.detach().numpy() > 0).astype('int64')  # P
    # target = target.numpy().astype('int64')  # T
    intersection = predict * (predict == target)  # TP

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))

    # areas of TN FP FN
    area_tp = area_inter[0]
    area_fp = area_pred[0] - area_inter[0]
    area_fn = area_lab[0] - area_inter[0]

    # area_union = area_pred + area_lab - area_inter
    assert area_tp <= (area_tp + area_fn + area_fp)
    return area_tp, area_fp, area_fn


class my_PD_FA(object):
    def __init__(self, ):
        self.reset()

    def update(self, pred, label):
        # max_pred= np.max(pred)
        # max_label = np.max(label)
        pred = np.array(pred)
        label = np.array(label)
        pred = pred / np.max(pred) # normalize output to 0-1
        label = label.astype(np.uint8)

        # analysis target number
        num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(label)
        #assert num_labels > 1
        if(num_labels <= 1):
            return

        # get masks and update background area and targets number
        back_mask = labels == 0
        tmp_back_area = np.sum(back_mask)
        self.background_area += tmp_back_area
        self.target_nums += (num_labels - 1)


        pred_binary = pred > 0.5

        # update false detection
        tmp_false_detect = np.sum(np.logical_and(back_mask, pred_binary))
        assert tmp_false_detect <= tmp_back_area
        self.false_detect += tmp_false_detect

        # update true detection, there maybe multiple targets
        for t in range(1, num_labels):
            target_mask = labels == t
            self.true_detect += np.sum(np.logical_and(target_mask, pred_binary)) > 0

    def get(self):
        FA = self.false_detect / self.background_area  #
        PD = self.true_detect / self.target_nums       #
        return PD,FA

    def get_all(self):
        return self.false_detect, self.background_area, self.true_detect, self.target_nums

    def reset(self):
        self.false_detect = 0
        self.true_detect = 0
        self.background_area = 0
        self.target_nums = 0