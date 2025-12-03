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

from data.sirst import NUDTDataset, IRSTD1kDataset, MDFADataset, SIRSTDataset
from utils.lr_scheduler import *
from utils.evaluation import SegmentationMetricTPFNFP, my_PD_FA
from utils.logger import setup_logger
from utils.utils import split_indices_by_mod
from net.DANnet import DNANet_withloss
from net.ACMnet import ASKCResUNet_withloss
from net.AGPCnet import AGPCNet_withloss


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
    parser.add_argument("--turn-num", type=str, default="0", help="choose pseudo label turn")
    parser.add_argument("--offset", type=int, default=0, help="offset of point label from center")
    parser.add_argument("--target-mix", type=int, default=0, help="whether to mix targets")
    parser.add_argument("--pseudo-label", type=int, default=1, help="whether to use pesudo label")
    parser.add_argument("--valset", type=int, default=0, help="whether to use real testset")
    parser.add_argument("--valset-mod", type=int, default=4, help="select valset from trainset by mod")
    parser.add_argument("--valset-rmd", type=int, default=0, help="select valset from trainset by mod")

    #
    # Training parameters
    #
    parser.add_argument("--batch-size", type=int, default=16, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--warm-up-epochs", type=int, default=0, help="warm up epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--lr-min", type=float, default=0.0, help="minimum learning rate")
    parser.add_argument("--gpu", type=str, default="0", help="GPU number")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--lr-scheduler", type=str, default="freepoly", help="learning rate scheduler")
    parser.add_argument("--label-thre", type=float, default=0.5, help="label vague threshold")
    parser.add_argument("--global-miou", type=float, default=0.5, help="global miou start")

    #
    # Net parameters
    #
    parser.add_argument("--net-name", type=str, default="rpcanet", help="net name: fcn")
    parser.add_argument("--model-path", type=str, default="", help="load model path")

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
    args.time_name = time.strftime("%Y%m%dT%H-%M-%S", time.localtime(time.time()))
    args.folder_name = "{}_{}_{}_{}".format(args.time_name, args.net_name, args.dataset, args.turn_num)
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

        # dataset
        using_pseudo_label = (args.pseudo_label == 1)
        if args.dataset == "nudt":
            trainset = NUDTDataset(base_dir=r"../IRSTD/NUDT-SIRST", mode="train", base_size=args.base_size, \
                                pseudo_label=using_pseudo_label, turn_num=args.turn_num, file_name="")
            valset = NUDTDataset(base_dir=r"../IRSTD/NUDT-SIRST", mode="test", base_size=args.base_size)
        # elif args.dataset == 'mdfa':
        #     trainset = MDFADataset(base_dir=r"../IRSTD/MDvsFA",mode="train", base_size=args.base_size, \
        #                         pseudo_label=using_pseudo_label, turn_num=args.turn_num, file_name="")
        #     valset = MDFADataset(base_dir=r"../IRSTD/MDvsFA",mode="test", base_size=args.base_size)
        elif args.dataset == 'sirst':
            trainset = SIRSTDataset(base_dir=r"../IRSTD/SIRST",mode="train", base_size=args.base_size, \
                                pseudo_label=using_pseudo_label, turn_num=args.turn_num, file_name="")
            valset = MDFADataset(base_dir=r"../IRSTD/SIRST",mode="test", base_size=args.base_size)
        elif args.dataset == "irstd1k":
            trainset = IRSTD1kDataset(base_dir=r"../IRSTD/IRSTD-1k", mode="train", base_size=args.base_size, \
                                pseudo_label=using_pseudo_label, turn_num=args.turn_num, file_name="")
            valset = IRSTD1kDataset(base_dir=r"../IRSTD/IRSTD-1k", mode="test", base_size=args.base_size)
        else:
            raise NotImplementedError
        if args.valset == 0:
            val_indices, train_indices = split_indices_by_mod(0, len(trainset)-1, args.valset_mod, args.valset_rmd)
            trainset, valset = Data.Subset(trainset, train_indices), Data.Subset(trainset, val_indices)

        self.train_data_loader = Data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, pin_memory=True, num_workers=1)
        self.val_data_loader = Data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=1)
        self.iter_per_epoch = len(self.train_data_loader)
        self.max_iter = args.epochs * self.iter_per_epoch

        ## GPU
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

        # model
        if args.net_name == "dnanet":
            self.net = DNANet_withloss(False, args.global_miou)
        elif args.net_name == "acmnet":
            self.net = ASKCResUNet_withloss(args.global_miou)
        elif args.net_name == "agpcnet":
            self.net = AGPCNet_withloss(args.global_miou)
        else:
            raise NotImplementedError
        
        self.net = self.net.to(self.device)

        ## lr scheduler
        self.scheduler = LR_Scheduler_Head(
            args.lr_scheduler, args.lr, args.epochs, len(self.train_data_loader), lr_step=10, min_lr=args.lr_min
        )

        ## optimizer
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
        self.best_FA = 1
        self.best_PD = 0
        self.eval_my_PD_FA = my_PD_FA()
        self.star_record_epoch_ratio = 0.5

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
        for epoch in range(args.epochs):
            for i, (data, label_) in enumerate(self.train_data_loader):
                data = data.to(self.device, non_blocking=True)
                label = label_[:,1:2] if self.args.pseudo_label == 1 else label_[:,0:1]
                label = label.to(self.device, non_blocking=True)
                label = (label >= self.args.label_thre).float()

                # _, softiouloss, class_loss, detail_loss, loss_128, _ = self.net(data, label)
                pred, softiouloss = self.net(data, label, epoch/args.epochs)
                
                # total_loss = softiouloss + class_loss + detail_loss + loss_128
                total_loss = softiouloss
                detail_loss = torch.tensor([0.], device=total_loss.device)
                loss_128 = torch.tensor([0.], device=total_loss.device)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

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

                # loss_softiou, loss_heatmap = torch.tensor([0.,]), torch.tensor([0.,])
                if self.iter_num % self.args.log_per_iter == 0:
                    self.logger.info(
                        base_log.format(
                            epoch + 1, args.epochs, self.iter_num % self.iter_per_epoch,
                            self.iter_per_epoch, self.optimizer.param_groups[0]["lr"],
                            total_loss.item(), softiouloss.item(), loss_128.item(), detail_loss.item(),
                            cost_string, eta_string,
                        )
                    )
                
                self.scheduler(self.optimizer, i, epoch, None, 1 - args.lr_min)
                
                if self.iter_num % self.iter_per_epoch == 0 and epoch/args.epochs > self.star_record_epoch_ratio:
                    self.net.eval()
                    self.validation()
                    self.net.train()
                    # self.scheduler(self.optimizer, i, epoch, None)


    def validation(self):
        self.metric.reset()
        self.eval_my_PD_FA.reset()
        base_log = "Data: {:s}, mIoU: {:.4f}/{:.4f}, PD: {:.4f}/{:.4f}, FA: {:.8f}/{:.8f}, F1: {:.4f}/{:.4f} "
        # base_log = "Data: {:s}, mIoU: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f}, Pd:{:.4f}, Fa:{:.8f} "
        for i, (img, label_) in enumerate(self.val_data_loader):
            if self.args.valset == 0:
                label = label_[:,1:2]
            else:
                label = label_[:,0:1]
            with torch.no_grad():
                pred, _ = self.net(img.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True))
            out_T = pred.cpu()

            label = (label >= self.args.label_thre).float()
            self.metric.update(label, out_T)
            self.eval_my_PD_FA.update(out_T, label)
        miou_all, prec_all, recall_all, fmeasure_all = self.metric.get()
        PD, FA = self.eval_my_PD_FA.get()

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
        if PD > self.best_PD:
            self.best_PD = PD
        if FA < self.best_FA:
            self.best_FA = FA

        # self.writer.add_scalar("Test/mIoU", miou_all, self.iter_num)
        # self.writer.add_scalar("Test/F1", fmeasure_all, self.iter_num)
        # self.writer.add_scalar("Best/mIoU", self.best_miou, self.iter_num)
        # self.writer.add_scalar("Best/Fmeasure", self.best_fmeasure, self.iter_num)

        self.logger.info(
            base_log.format(self.args.dataset, miou_all, self.best_miou, PD, self.best_PD, FA, self.best_FA, fmeasure_all, self.best_fmeasure)
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
    trainer.load_model(args.model_path)
    trainer.training()

    # print('Best mIoU: %.5f, Best Fmeasure: %.5f\n\n' % (trainer.best_miou, trainer.best_fmeasure))
