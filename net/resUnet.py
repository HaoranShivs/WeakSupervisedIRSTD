import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from net.basenet import Resconv, ShallowFeatureExtractor, DeepFeatureExtractor, UpScaler, Conv2d_Bn_Relu, DetectNet1, BaseNet3, ResBlock
from net.twotasknet import ConvT2d_Bn_Relu
from utils.loss import SoftLoULoss, Detail_loss


