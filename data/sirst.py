import torch

import torch.utils.data as Data
import torchvision.transforms as transforms

import cv2
import os
import os.path as osp

from data.utils import RandomResize, Augment_transform, mask2point


__all__ = ["SirstAugDataset", "IRSTD1kDataset", "NUDTDataset"]


class IRSTD1kDataset(Data.Dataset):
    """
    Return: Single channel
    """

    def __init__(
        self,
        base_dir=r"W:/DataSets/Infraid_datasets/IRSTD-1k",
        mode="train",
        base_size=256,
        pt_label=False,
        offset = 0,
        pesudo_label=False,
        preded_label=False,
        augment=True,
        turn_num='',
        target_mix = False,
        file_name = '',
        cfg=None
    ):
        assert mode in ["train", "test"]

        if mode == "train":
            self.data_dir = osp.join(base_dir, "trainval")
        elif mode == "test":
            self.data_dir = osp.join(base_dir, "test")
        else:
            raise NotImplementedError
        
        self.mode = mode
        self.cfg = cfg
        self.base_size = base_size
        self.pt_label = pt_label
        self.offset = offset
        self.pesudo_label = pesudo_label
        self.preded_label = preded_label
        self.aug = augment
        self.turn_num = turn_num
        self.target_mix = target_mix
        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images' + file_name)):
            if filename.endswith("png"):
                self.names.append(filename)

        self.aug_transformer = Augment_transform(base_size, mode) if augment else Augment_transform(base_size, "test")
        # self.gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.8, 1.0)) if target_mix else None
        # self.target_trans = transforms.Compose([
        #     transforms.RandomAffine(degrees=180, translate=(0.3, 0.3)),
        #     transforms.RandomHorizontalFlip()])  # 随机水平翻转
        # self.gaussian_filter3 = transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.5)) if target_mix else None
        # self.gaussian_filter9 = transforms.GaussianBlur(kernel_size=9, sigma=(0.3, 0.5)) if target_mix else None

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, "images", name)
        pesudo_label_path = osp.join(self.data_dir, f'pixel_pseudo_label{self.turn_num}', name)
        preded_label_path = osp.join(self.data_dir, f'preded_label/{self.turn_num}', name)
        label_path = osp.join(self.data_dir, "masks", name)

        img, mask= cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        img, mask= torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(mask).unsqueeze(0).float()
        
        if self.pesudo_label:
            pesudo_label = cv2.imread(pesudo_label_path, 0)
            pesudo_label = torch.from_numpy(pesudo_label).unsqueeze(0).float()
            mask = torch.cat([mask, pesudo_label], dim=0)
        if self.preded_label:
            preded_label = cv2.imread(preded_label_path, 0)
            preded_label = torch.from_numpy(preded_label).unsqueeze(0).float()
            mask = torch.cat([mask, preded_label], dim=0)
        
        img, mask = self.aug_transformer(img, mask)

        img, mask = img / 255.0, mask / 255.0

        if self.pt_label:
            pt_label = mask2point(mask[0].unsqueeze(0), img, self.offset)
            mask[0] = pt_label

        return img, mask

    def __len__(self):
        return len(self.names)


class NUDTDataset(Data.Dataset):
    """
    Return: Single channel
    """
    def __init__(
        self,
        base_dir=r"W:/DataSets/Infraid_datasets/NUDT-SIRST",
        mode="train",
        base_size=256,
        pt_label=False,
        offset = 0,
        pesudo_label=False,
        preded_label=False,
        augment=True,
        turn_num='',
        target_mix = False,
        file_name = '',
        cfg=None
    ):
        assert mode in ["train", "test"]

        if mode == "train":
            self.data_dir = osp.join(base_dir, "trainval")
        elif mode == "test":
            self.data_dir = osp.join(base_dir, "test")
        else:
            raise NotImplementedError
        self.mode = mode 
        self.base_size = base_size
        self.cfg = cfg
        self.pt_label = pt_label
        self.offset = offset
        self.pesudo_label = pesudo_label
        self.preded_label = preded_label
        self.aug = augment
        self.turn_num = turn_num
        self.target_mix = target_mix
        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images' + file_name)):
            if filename.endswith("png"):
                self.names.append(filename)

        self.aug_transformer = Augment_transform(base_size, mode) if augment else Augment_transform(base_size, "test")
        # self.gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.8, 1.0)) if target_mix else None
        # self.target_trans = transforms.Compose([
        #     transforms.RandomAffine(degrees=180, translate=(0.3, 0.3)),
        #     transforms.RandomHorizontalFlip()])  # 随机水平翻转
        # self.gaussian_filter3 = transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.5)) if target_mix else None
        # self.gaussian_filter9 = transforms.GaussianBlur(kernel_size=9, sigma=(0.3, 0.5)) if target_mix else None

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, "images", name)
        pesudo_label_path = osp.join(self.data_dir, f'pixel_pseudo_label{self.turn_num}', name)
        preded_label_path = osp.join(self.data_dir, f'preded_label/{self.turn_num}', name)
        label_path = osp.join(self.data_dir, "masks", name)

        img, mask= cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        img, mask= torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(mask).unsqueeze(0).float()
        
        if self.pesudo_label:
            pesudo_label = cv2.imread(pesudo_label_path, 0)
            pesudo_label = torch.from_numpy(pesudo_label).unsqueeze(0).float()
            mask = torch.cat([mask, pesudo_label], dim=0)
        if self.preded_label:
            preded_label = cv2.imread(preded_label_path, 0)
            preded_label = torch.from_numpy(preded_label).unsqueeze(0).float()
            mask = torch.cat([mask, preded_label], dim=0)
        
        img, mask = self.aug_transformer(img, mask)

        img, mask = img / 255.0, mask / 255.0

        if self.pt_label:
            pt_label = mask2point(mask[0].unsqueeze(0), img, self.offset)
            mask[0] = pt_label

        return img, mask
    
    def __len__(self):
        return len(self.names)
    

# class SirstAugDataset(Data.Dataset):
#     '''
#     Return: Single channel
#     '''
#     def __init__(self, base_dir=r'/Users/tianfangzhang/Program/DATASETS/sirst_aug',
#                  mode='train', base_size=256):
#         assert mode in ['train', 'test']

#         if mode == 'train':
#             self.data_dir = osp.join(base_dir, 'trainval')
#         elif mode == 'test':
#             self.data_dir = osp.join(base_dir, 'test')
#         else:
#             raise NotImplementedError

#         self.base_size = base_size

#         self.names = []
#         for filename in os.listdir(osp.join(self.data_dir, 'images')):
#             if filename.endswith('png'):
#                 self.names.append(filename)
#         self.tranform = augumentation()
#         # self.transform = transforms.Compose([
#         #     transforms.ToTensor(),
#         #     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
#         # ])

#     def __getitem__(self, i):
#         name = self.names[i]
#         img_path = osp.join(self.data_dir, 'images', name)
#         label_path = osp.join(self.data_dir, 'masks', name)

#         img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
#         img, mask = self.tranform(img, mask)
#         img = img.reshape(1, self.base_size, self.base_size) / 255.
#         if np.max(mask) > 0:
#             mask = mask.reshape(1, self.base_size, self.base_size) / np.max(mask)
#         else:
#             mask = mask.reshape(1, self.base_size, self.base_size)
#         # row, col = img.shape
#         # img = img.reshape(1, row, col) / 255.
#         # if np.max(mask) > 0:
#         #     mask = mask.reshape(1, row, col) / np.max(mask)
#         # else:
#         #     mask = mask.reshape(1, row, col)

#         img = torch.from_numpy(img).type(torch.FloatTensor)
#         mask = torch.from_numpy(mask).type(torch.FloatTensor)
#         return img, mask

#     def __len__(self):
#         return len(self.names)