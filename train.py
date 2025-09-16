import os
import os.path as osp
import time
import datetime
from argparse import ArgumentParser

import yaml
import numpy as np
import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

# from models import get_model
from data.sirst import NUDTDataset, IRSTD1kDataset
# from dataprocess.croped_sirst import Crop_IRSTD1kDataset, Crop_NUDTDataset
# from net.basenet import BaseNet1, BaseNet2, BaseNet3, BaseNet4, BaseNetWithLoss, LargeBaseNet, LargeBaseNet2, GaussNet, GaussNet2, GaussNet3, GaussNet4, SigmoidNet
# from net.basenet import BaseNet4, BaseNetWithLoss
# from net.twotasknet import Heatmap_net, LocalSegment, HeatMaptoImg, UnitLabels, TwoTaskNetWithLoss
from net.attentionnet import attenMultiplyUNet_withloss
from utils.loss import SoftLoULoss, ImageRecoverLoss, Heatmap_SoftIoU, Heatmap_MSE
from utils.lr_scheduler import *
from utils.evaluation import SegmentationMetricTPFNFP, my_PD_FA
from utils.logger import setup_logger
from net.DANnet import DNANet_withloss
from net.ACMnet import ASKCResUNet_withloss


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description="Implement of BaseNet")

    #
    # Dataset parameters
    #
    parser.add_argument("--base-size", type=int, default=256, help="base size of images")
    parser.add_argument("--crop-size", type=int, default=256, help="crop size of images")
    parser.add_argument("--dataset", type=str, default="nudt", help="choose datasets")

    #
    # Training parameters
    #

    parser.add_argument("--batch-size", type=int, default=32, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--warm-up-epochs", type=int, default=0, help="warm up epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gpu", type=str, default="0", help="GPU number")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--lr-scheduler", type=str, default="poly", help="learning rate scheduler")

    #
    # Net parameters
    #
    parser.add_argument("--net-name", type=str, default="dnanet", help="net name: fcn")
    parser.add_argument("--model-path", type=str, default="", help="load model path")
    parser.add_argument("--model-path1", type=str, default="", help="load model path")
    parser.add_argument("--model-path2", type=str, default="", help="load model path")
    # Rank parameters
    #
    # parser.add_argument('--rank', type=int, default=8,
    #                     help='rank number')

    #
    # Save parameters
    #
    parser.add_argument("--save-iter-step", type=int, default=10, help="save model per step iters")
    parser.add_argument("--log-per-iter", type=int, default=1, help="interval of logging")
    parser.add_argument("--base-dir", type=str, default="./result", help="saving dir")

    #
    # Configuration
    #
    parser.add_argument("--cfg-path", type=str, default="./cfg.yaml", help="path of cfg file")

    args = parser.parse_args()

    # Save folders
    # args.base_dir = r'D:\WFY\dun_irstd\result'
    args.time_name = time.strftime("%Y%m%dT%H-%M-%S", time.localtime(time.time()))
    args.folder_name = "{}_{}_{}".format(args.time_name, args.net_name, args.dataset)
    args.save_folder = osp.join(args.base_dir, args.folder_name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # seed
    if args.seed != 0:
        set_seeds(args.seed)

    # logger
    args.logger = setup_logger("BaseNet test", args.save_folder, 0, filename="log.txt", mode="a")
    return args


def set_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.iter_num = 0

        ## cfg file
        with open(args.cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        with open(osp.join(self.args.save_folder, "cfg.yaml"), "w", encoding="utf-8") as file:
            yaml.dump(self.cfg, file, allow_unicode=True)

        # dataset
        if args.dataset == "nudt":
            trainset = NUDTDataset(
                base_dir=r"W:/DataSets/ISTD/NUDT-SIRST", mode="train", base_size=args.base_size, cfg=self.cfg, pesudo_label=False,
                augment=False
            )
            valset = NUDTDataset(
                base_dir=r"W:/DataSets/ISTD/NUDT-SIRST", mode="train", base_size=args.base_size, cfg=self.cfg
            )
        # elif args.dataset == 'sirstaug':
        #     trainset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
        #                                mode='train', base_size=args.base_size)  # base_dir=r'E:\ztf\datasets\sirst_aug'
        #     valset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
        #                              mode='test', base_size=args.base_size)  # base_dir=r'E:\ztf\datasets\sirst_aug'
        elif args.dataset == "irstd1k":
            trainset = IRSTD1kDataset(
                base_dir=r"W:/DataSets/ISTD/IRSTD-1k", mode="train", base_size=args.base_size, cfg=self.cfg, pesudo_label=False,
                augment=False
            )
            valset = IRSTD1kDataset(
                base_dir=r"W:/DataSets/ISTD/IRSTD-1k", mode="train", base_size=args.base_size, cfg=self.cfg
            )
        else:
            raise NotImplementedError

        self.train_data_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        self.val_data_loader = Data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)
        self.iter_per_epoch = len(self.train_data_loader)
        self.max_iter = args.epochs * self.iter_per_epoch

        ## GPU
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

        # model
        if args.net_name == "dnanet":
            self.net = DNANet_withloss(False, 0.75)
        elif args.net_name == "acmnet":
            self.net = ASKCResUNet_withloss(0.75)
        elif args.net_name == "agpcnet":
            self.net = TwoTaskNetWithLoss()
        

        # net = BaseNet4(1, self.cfg)
        # loss_fn = SoftLoULoss()
        # self.net = attenMultiplyUNet_withloss(self.cfg, False)

        # self.net.apply(self.weight_init)
        self.net = self.net.to(self.device)

        ## criterion
        # self.heatmap_softiou = Heatmap_SoftIoU(self.cfg)
        # self.heatmap_mse = Heatmap_MSE(self.cfg)
        # self.softiou = SoftLoULoss()

        ## lr scheduler
        self.scheduler = LR_Scheduler_Head(
            args.lr_scheduler, args.lr, args.epochs, len(self.train_data_loader), lr_step=10
        )

        ## optimizer
        # self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.learning_rate,
        #                                  momentum=0.9, weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)

        ## evaluation metrics
        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.best_miou = 0
        self.best_fmeasure = 0
        self.best_prec = 0
        self.best_recall = 0
        self.eval_loss = 0  # tmp values
        self.miou = 0
        self.fmeasure = 0
        self.eval_my_PD_FA = my_PD_FA()

        ## SummaryWriter
        self.writer = SummaryWriter(log_dir=args.save_folder)
        self.writer.add_text(args.folder_name, "Args:%s, " % args)

        ## log info
        self.logger = args.logger
        self.logger.info(args)
        self.logger.info("Using device: {}".format(self.device))

    def training(self):
        # training step
        start_time = time.time()
        base_log = "Epoch-Iter: [{:03d}/{:03d}]-[{:03d}/{:03d}]  || Lr: {:.6f} ||  Loss: {:.4f}={:.4f}+{:.4f}+{:.4f} || " \
                   "Cost Time: {} || Estimated Time: {}"
        self.net.eval()
        for epoch in range(args.epochs):
            for i, (data, label) in enumerate(self.train_data_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                label_b = (label > self.cfg["label_vague_threshold"]).type(torch.float32)

                # with torch.no_grad():
                # _, softIoU_loss, class_loss, detail_loss, loss_128 = self.net(data, label)
                pred, softIoU_loss = self.net(data, label_b, epoch/args.epochs, label)
                # total_loss = softIoU_loss + class_loss + detail_loss + loss_128
                total_loss = softIoU_loss
                detail_loss = torch.tensor([0.], device=total_loss.device)
                class_loss = torch.tensor([0.], device=total_loss.device)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # for name, param in self.net.named_parameters():
                #     if "net.linear" in name:
                #         print(f"Gradient for {name}: {param.grad}")

                self.iter_num += 1

                cost_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                eta_seconds = ((time.time() - start_time) / self.iter_num) * (self.max_iter - self.iter_num)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                self.writer.add_scalar('Train Loss/Loss All', np.mean(total_loss.item()), self.iter_num)
                # self.writer.add_scalar("Train Loss/Loss SoftIoU", np.mean(loss_softiou.item()), self.iter_num)
                # self.writer.add_scalar('Train Loss/Loss MSE', np.mean(loss_mse.item()), self.iter_num)
                self.writer.add_scalar(
                    "Learning rate/", trainer.optimizer.param_groups[0]["lr"], self.iter_num
                )

                if self.iter_num % self.args.log_per_iter == 0:
                    self.logger.info(
                        base_log.format(
                            epoch + 1,
                            args.epochs,
                            self.iter_num % self.iter_per_epoch,
                            self.iter_per_epoch,
                            self.optimizer.param_groups[0]["lr"],
                            total_loss.item(), softIoU_loss.item(), class_loss.item(), detail_loss.item(),
                            cost_string,
                            eta_string,
                        )
                    )

                if self.iter_num % self.iter_per_epoch == 0:
                    self.net.eval()
                    self.validation()
                    self.net.train()
                    self.scheduler(self.optimizer, i, epoch, None)


    def validation(self):
        self.metric.reset()
        # self.eval_my_PD_FA.reset()
        base_log = "Data: {:s}, mIoU: {:.4f}/{:.4f}, prec: {:.4f}/{:.4f}, recall: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f} "
        # base_log = "Data: {:s}, mIoU: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f}, Pd:{:.4f}, Fa:{:.8f} "
        for i, (data, labels) in enumerate(self.val_data_loader):
            with torch.no_grad():
                data = data.repeat(1, 3, 1, 1)  # for DNANet
                # noise = torch.zeros((data.shape[0], 1, 32, 32), device=data.device)
                # pred, _, _, _, _ = self.net.net(data.to(self.device))
                pred, _ = self.net.net(data.to(self.device))
                pred = pred[-1]
            out_T = pred.cpu() 

            # loss_softiou = self.softiou(out_T, labels)
            # loss_mse = self.mse(out_D, data)
            # gamma = torch.Tensor([0.1]).to(self.device)
            # loss_all = loss_softiou + torch.mul(gamma, loss_mse)
            
            labels = (labels > self.cfg["label_vague_threshold"]).type(torch.float32)
            self.metric.update(labels, out_T)
        miou_all, prec_all, recall_all, fmeasure_all = self.metric.get()

        torch.save(self.net.state_dict(), osp.join(self.args.save_folder, "latest.pkl"))
        if miou_all > self.best_miou:
            self.best_miou = miou_all
            torch.save(self.net.state_dict(), osp.join(self.args.save_folder, "best.pkl"))
        if fmeasure_all > self.best_fmeasure:
            self.best_fmeasure = fmeasure_all
        if prec_all > self.best_prec:
            self.best_prec = prec_all
        if recall_all > self.best_recall:
            self.best_recall = recall_all

        # print(miou, self.best_miou, fmeasure, self.best_fmeasure)

        self.writer.add_scalar("Test/mIoU", miou_all, self.iter_num)
        self.writer.add_scalar("Test/F1", fmeasure_all, self.iter_num)
        self.writer.add_scalar("Best/mIoU", self.best_miou, self.iter_num)
        self.writer.add_scalar("Best/Fmeasure", self.best_fmeasure, self.iter_num)

        self.logger.info(
            base_log.format(self.args.dataset, miou_all, self.best_miou, prec_all, self.best_prec, recall_all, self.best_recall, fmeasure_all, self.best_fmeasure)
        )

    def load_model(self, model_path: str = "", model_path1: str = "", model_path2: str = ""):
        if model_path != "":
            model_path = osp.join(model_path, "best.pkl")
            self.net.load_state_dict(torch.load(model_path))
        if model_path1 != "":
            model_path1 = osp.join(model_path1, "best.pkl")
            self.net.net_heatmap.load_state_dict(torch.load(model_path1))
        if model_path2 != "":
            model_path2 = osp.join(model_path2, "best.pkl")
            self.net.net_localseg.load_state_dict(torch.load(model_path2))


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)
    trainer.load_model(args.model_path, args.model_path1, args.model_path2)
    trainer.training()

    # print('Best mIoU: %.5f, Best Fmeasure: %.5f\n\n' % (trainer.best_miou, trainer.best_fmeasure))
