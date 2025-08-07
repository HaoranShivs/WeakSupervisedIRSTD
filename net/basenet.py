import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils import hide_channels, gaussian_kernel


class GaussianConv2d(nn.Module):
    def __init__(self, in_channel, kernel_size, sigma=0.6):
        super(GaussianConv2d, self).__init__()
        padding = kernel_size // 2
        self.tempreture = nn.Parameter(torch.randn((1, in_channel, 1, 1)) + 2.5)
        self.bias = nn.Parameter(torch.randn((1, in_channel, 1, 1)))

        # 创建高斯核
        kernel = gaussian_kernel(in_channel, kernel_size, sigma)  # (channel, channel, kernelsize, kernelsize)
        # kernel = kernel.unsqueeze(0).unsqueeze(0)  # 添加通道维度

        # 创建Gauss卷积层
        self.atten = nn.Conv2d(in_channel, in_channel, kernel_size, padding=padding, bias=False)
        self.atten.weight.data = kernel  # 设置权重
        self.atten.weight.requires_grad = False  # 不需要梯度更新

        # # Gauss saliency
        # self.sigmoid = nn.Sigmoid()

    def forward(self, f):
        # img = self.recov(input) # (B, 1, H, W)
        # B, C, H, W = f.shape
        # f = f.view(B * C, 1, H, W)
        atten = self.atten(f)  # (-1,1)
        # atten = atten.view(B, C, H, W)
        # atten = self.sigmoid(atten * self.tempreture + self.bias) * 2 - 1
        atten = torch.abs(atten * self.tempreture) + self.bias
        return atten  # return positive attention (B, C, H, W)


class GaussianConv2d2(nn.Module):
    def __init__(self, in_channel, kernel_size, sigma=0.6):
        super(GaussianConv2d2, self).__init__()
        padding = kernel_size // 2
        self.kernel_size = kernel_size

        # 参数
        self.sigma = nn.Parameter(torch.ones((kernel_size, kernel_size))*sigma)
        self.bias = nn.Parameter(torch.randn((kernel_size, kernel_size))-0.1)

        # 创建Gauss卷积层
        self.atten = nn.Conv2d(1, 9, kernel_size, padding=padding, bias=False)
        
        self.conv = Conv2d_Bn_Relu(in_channel * 2, in_channel, 3, 1, 1)

        # # Gauss saliency
        # self.sigmoid = nn.Sigmoid()

    def __set_kernel__(self):
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                x = torch.arange(0, self.kernel_size, 1, dtype=torch.float32, device=self.sigma.device) - i
                y = torch.arange(0, self.kernel_size, 1, dtype=torch.float32, device=self.sigma.device) - j
                kernel_1d_x = torch.exp(-(x**2) / (2 * self.sigma[i,j]**2))
                kernel_1d_y = torch.exp(-(y**2) / (2 * self.sigma[i,j]**2))
                kernel_2d = torch.outer(kernel_1d_x, kernel_1d_y)
                # 归一化
                kernel_2d /= kernel_2d.sum()
                # 产生正负值
                kernel_2d -= self.bias[i, j]
                # 赋值
                self.atten.weight.data[i*self.kernel_size+j, 0] = kernel_2d

    def forward(self, f):
        # img = self.recov(input) # (B, 1, H, W)
        self.__set_kernel__()
        B, C, H, W = f.shape
        f = f.view(B * C, 1, H, W)
        score = self.atten(f)  # (-1,1), (B*C, kernel_num, H, W)
        score = torch.sum(score, dim=1, keepdim=True) # (B*C, 1, H, W)
        score = score.view(B, C, H, W)
        atten = torch.cat((score, f.view(B, C, H, W)), dim=1)
        atten = self.conv(atten)    # (B, C, H, W)

        # atten = self.sigmoid(atten * self.tempreture + self.bias) * 2 - 1
        # atten = torch.abs(atten)
        return atten  # return positive attention (B, C, H, W)


class Conv2d_Bn_Relu(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        super(Conv2d_Bn_Relu, self).__init__()
        self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv2d(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel: int = 32,
        out_channel: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv3 = Conv2d_Bn_Relu(in_channel, out_channel, 1, 1)

    def forward(self, inputs):
        x = self.relu(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))
        # x = self.relu(x)
        # x = self.bn3(self.conv2d3(x))
        x = x + self.conv3(inputs)
        x = self.relu(x)
        return x


class Gauss_ResBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size: int = 3, stride: int = 1, gauss_kernelsize=3, gauss_sigma=0.6
    ):
        super(Gauss_ResBlock, self).__init__()

        self.conv1 = Conv2d_Bn_Relu(in_channel, out_channel, kernel_size, stride, 1)
        # attention
        # self.gauss = GaussianConv2d(out_channel, gauss_kernelsize, gauss_sigma)
        self.gauss = GaussianConv2d2(gauss_kernelsize, gauss_sigma)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
        self.conv3 = Conv2d_Bn_Relu(in_channel, out_channel, 1, 1)

    def forward(self, shallow_feature):
        f = self.conv1(shallow_feature)
        atten = self.gauss(f)
        deep_feature = self.bn2(self.conv2(f + atten))
        deep_feature = self.relu2(deep_feature + self.conv3(shallow_feature))
        return deep_feature


class Resconv(nn.Module):
    def __init__(self, in_channel=1, out_channel=32, down_sampler=None):
        super(Resconv, self).__init__()

        self.block1 = ResBlock(in_channel, int(out_channel / 4))
        self.block2 = ResBlock(int(out_channel / 4), int(out_channel / 2))
        self.block3 = ResBlock(int(out_channel / 2), out_channel)

        self.down_sampler = down_sampler

    def forward(self, img):
        """
        args:
            img(torch.tensor): (B, C1, H, W)
        output:
            reconstructed_img(torch.tensor): ( B, C2, H, W)
        """
        x = self.block1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.down_sampler(x) if self.down_sampler is not None else x
        return x


class Gauss_Resconv(nn.Module):
    def __init__(self, in_channel, out_channel, gauss_kernelsize=3, gauss_sigma=0.6, down_sampler=None):
        super(Gauss_Resconv, self).__init__()

        self.block1 = Gauss_ResBlock(in_channel, int(out_channel / 4), 3, 1, gauss_kernelsize, gauss_sigma)
        self.block2 = Gauss_ResBlock(int(out_channel / 4), int(out_channel / 2), 3, 1, gauss_kernelsize, gauss_sigma)
        self.block3 = Gauss_ResBlock(int(out_channel / 2), out_channel, 3, 1, gauss_kernelsize, gauss_sigma)

        self.down_sampler = down_sampler

    def forward(self, img):
        """
        args:
            img(torch.tensor): (B, C1, H, W)
        output:
            reconstructed_img(torch.tensor): ( B, C2, H, W)
        """
        x = self.block1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.down_sampler(x) if self.down_sampler is not None else x
        return x


class ShallowFeatureExtractor(nn.Module):  
    def __init__(
        self,
        in_channel: int = 32,
        out_channel: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        downsampler=None,
    ):
        super(ShallowFeatureExtractor, self).__init__()
        self.conv1 = Conv2d_Bn_Relu(in_channel, out_channel, kernel_size, stride, 1)
        self.conv2 = Conv2d_Bn_Relu(out_channel, out_channel, kernel_size, stride, 1)
        self.downsampler = downsampler

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.downsampler(x) if self.downsampler is not None else x
        return x


class DeepFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channel: int = 32,
        out_channel: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super(DeepFeatureExtractor, self).__init__()
        self.conv1 = Conv2d_Bn_Relu(in_channel, out_channel, kernel_size, stride, 1)
        self.conv2 = Conv2d_Bn_Relu(out_channel, out_channel, 1, 1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


class Conv2d_with_Gauss(nn.Module):
    def __init__(self, in_channel, out_channel, gauss_sigma=0.6, downsampler=None):
        super(Conv2d_with_Gauss, self).__init__()

        # attention
        self.gauss = GaussianConv2d(in_channel, 3, gauss_sigma)
        self.conv = ShallowFeatureExtractor(in_channel, out_channel, 3, 1, downsampler)

    def forward(self, shallow_feature):
        atten = self.gauss(shallow_feature)
        deep_feature = self.conv(shallow_feature * atten)
        return deep_feature


class Conv2d_with_Gauss2(nn.Module):
    def __init__(self, in_channel, out_channel, gauss_sigma=0.6, downsampler=None):
        super(Conv2d_with_Gauss2, self).__init__()

        self.conv1 = Conv2d_Bn_Relu(in_channel, out_channel, 3, 1, 1)
        # attention
        self.gauss = GaussianConv2d(out_channel, 3, gauss_sigma)
        self.conv2 = Conv2d_Bn_Relu(out_channel, out_channel, 3, 1, 1)
        self.downsampler = downsampler

    def forward(self, shallow_feature):
        f = self.conv1(shallow_feature)
        atten = self.gauss(f)
        deep_feature = self.conv2(f * atten)
        deep_feature = self.downsampler(deep_feature) if self.downsampler is not None else deep_feature
        return deep_feature


class Conv2d_with_Gauss3(nn.Module):
    def __init__(self, in_channel, out_channel, gauss_kernelsize=3, gauss_sigma=0.6, downsampler=None):
        super(Conv2d_with_Gauss3, self).__init__()

        self.conv1 = Conv2d_Bn_Relu(in_channel, out_channel, 3, 1, 1)
        # attention
        # self.gauss = GaussianConv2d(out_channel, gauss_kernelsize, gauss_sigma)
        self.gauss = GaussianConv2d2(out_channel, gauss_kernelsize, gauss_sigma)
        self.conv2 = Conv2d_Bn_Relu(out_channel, out_channel, 3, 1, 1)
        self.downsampler = downsampler

    def forward(self, shallow_feature):
        f = self.conv1(shallow_feature)
        atten = self.gauss(f)
        deep_feature = self.conv2(f + atten)
        deep_feature = self.downsampler(deep_feature) if self.downsampler is not None else deep_feature
        return deep_feature


class Conv2dwithSigmoid(nn.Module):
    def __init__(self, in_channel, out_channel, downsampler=None):
        super(Conv2dwithSigmoid, self).__init__()

        self.conv1 = Conv2d_Bn_Relu(in_channel, out_channel, 3, 1, 1)
        # attention
        self.atten = nn.Sigmoid()
        self.conv2 = Conv2d_Bn_Relu(out_channel, out_channel, 3, 1, 1)
        self.downsampler = downsampler

    def forward(self, shallow_feature):
        f = self.conv1(shallow_feature)
        atten = self.atten(f)
        deep_feature = self.conv2(f * atten)
        deep_feature = self.downsampler(deep_feature) if self.downsampler is not None else deep_feature
        return deep_feature


class UpScaler(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, is_lastone=False):
        super(UpScaler, self).__init__()
        self.proj1 = DeepFeatureExtractor(in_channel, in_channel, kernel_size, 1)
        self.convT = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, 2, 1, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.proj2 = DeepFeatureExtractor(out_channel, out_channel, kernel_size, 1) if not is_lastone else None

    def forward(self, f_deep, f_shallow=None):
        x1 = self.proj1(f_deep)
        x1 = self.relu(self.bn(self.convT(x1)))
        if self.proj2 is None:
            return x1
        x2 = self.proj2(f_shallow)
        res = torch.max(x1, x2)
        return res


class UpScaler_cat(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(UpScaler_cat, self).__init__()
        self.conv1 = nn.Sequential(Conv2d_Bn_Relu(out_channel+in_channel, out_channel+in_channel, 1),
                                   Conv2d_Bn_Relu(out_channel+in_channel, out_channel+in_channel, 1))
        self.conv2 = ResBlock(out_channel+in_channel, out_channel)
        self.conv_residual = Conv2d_Bn_Relu(out_channel+in_channel, out_channel+in_channel, 1)

    def forward(self, f_deep, f_shallow=None):
        f_deep = F.interpolate(f_deep, scale_factor=2, mode='nearest')
        f = torch.concatenate((f_deep, f_shallow), dim=1)
        f = f + self.conv1(f)
        f = self.conv2(f)
        return f


class UpScaler_compare(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(UpScaler_compare, self).__init__()
        self.conv1 = ResBlock(in_channel, out_channel)
        self.conv2 = ResBlock(out_channel, out_channel)

    def forward(self, f_deep, f_shallow=None):
        # upscale and decrease channel
        f_deep = F.interpolate(f_deep, scale_factor=2, mode='nearest')
        f_deep = self.conv1(f_deep)
        f_diff = F.sigmoid(torch.sum((f_deep - f_shallow).pow(2), dim=1)) * 2 - 1
        f_diff = f_diff.unsqueeze(1)
        f = f_shallow * f_diff + f_deep * (1 - f_diff)
        f = self.conv2(f)
        return f


class UpScaler_compare2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(UpScaler_compare2, self).__init__()
        self.conv1 = ResBlock(in_channel, out_channel)
        self.conv2 = ResBlock(out_channel, out_channel)

    def forward(self, f_deep, f_shallow=None):
        # upscale and decrease channel
        f_deep = F.interpolate(f_deep, scale_factor=2, mode='nearest')
        f_deep = self.conv1(f_deep)
        f_diff = F.sigmoid((f_deep - f_shallow).pow(2)) * 2 - 1
        f = f_shallow * f_diff + f_deep * (1 - f_diff)
        f = self.conv2(f)
        return f


class MultiScaleFeatureNet(nn.Module):
    def __init__(self, in_channel, out_channel_list, atten_choice=0, cfg=None, downsampler=None):
        super(MultiScaleFeatureNet, self).__init__()

        out_channel_list.insert(0, in_channel)

        pyramid_layers = []
        for i in range(1, len(out_channel_list)):
            if atten_choice == 1:
                layer = Conv2d_with_Gauss3(
                    out_channel_list[i - 1],
                    out_channel_list[i],
                    cfg["gauss_kernelsize"],
                    cfg["gauss_sigma"],
                    downsampler,
                )
            elif atten_choice == 2:
                layer = Conv2dwithSigmoid(
                    out_channel_list[i - 1],
                    out_channel_list[i],
                    downsampler,
                )
            else:
                layer = ShallowFeatureExtractor(out_channel_list[i - 1], out_channel_list[i], 3, 1, downsampler)
            pyramid_layers.append(layer)

        self.pyramid = nn.ModuleList(pyramid_layers)

    def forward(self, shallow_feature):
        """
        args:
            shallow_feature(torch.tensor): (B, C1, H, W)
        output:
            (...,H/2, W/2)
            (...,H/4, W/4)
            (...,H/8, W/8)
            ...
        """
        outputs = []
        x = shallow_feature
        for i in self.pyramid:
            x = i(x)
            outputs.append(x)
        return outputs


class FusionNet_plus(nn.Module):
    def __init__(self, in_channel_list=None, out_channel=256):
        super(FusionNet_plus, self).__init__()
        self.comp1 = DeepFeatureExtractor(in_channel_list[0], out_channel, 3, 1)
        self.comp2 = DeepFeatureExtractor(in_channel_list[1], out_channel, 3, 1)
        self.comp3 = DeepFeatureExtractor(in_channel_list[2], out_channel, 3, 1)
        self.comp4 = DeepFeatureExtractor(in_channel_list[3], out_channel, 3, 1)

    def forward(self, feature1, feature2, feature3, feature4):
        f4 = self.comp4(feature4).repeat(1, 1, 8, 8)
        f1 = self.comp1(feature1)
        f2 = self.comp2(feature2).repeat(1, 1, 2, 2)
        f3 = self.comp3(feature3).repeat(1, 1, 4, 4)

        mixf = f1 + f2 + f3 + f4

        return mixf


class FusionNet_cat(nn.Module):
    def __init__(self, in_channel_list=None, out_channel=256):
        super(FusionNet_cat, self).__init__()
        self.cf1 = DeepFeatureExtractor(sum(in_channel_list[:1]), sum(in_channel_list[:1]), 3, 1)
        self.cf2 = DeepFeatureExtractor(sum(in_channel_list[:2]), sum(in_channel_list[:2]), 3, 1)
        self.cf3 = DeepFeatureExtractor(sum(in_channel_list[:3]), sum(in_channel_list[:3]), 3, 1)
        self.cf4 = DeepFeatureExtractor(sum(in_channel_list[:4]), sum(in_channel_list[:4]), 3, 1)

        self.transform = Conv2d_Bn_Relu(sum(in_channel_list[:4]), out_channel, 1, 1)

    def forward(self, feature1, feature2, feature3, feature4):
        f1 = self.cf1(feature1)
        f2 = torch.cat((f1, feature2.repeat(1, 1, 2, 2)), dim=1)
        f2 = self.cf2(f2)
        f3 = torch.cat((f2, feature3.repeat(1, 1, 4, 4)), dim=1)
        f3 = self.cf3(f3)
        f4 = torch.cat((f3, feature4.repeat(1, 1, 8, 8)), dim=1)
        f4 = self.cf4(f4)

        fusion_feature = self.transform(f4)

        return fusion_feature


class FusionNet_upscale(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(FusionNet_upscale, self).__init__()

        SRer_layers = []
        for i in range(len(in_channel_list) - 1, 1, -1):
            SRer_layers.append(UpScaler(in_channel_list[i], in_channel_list[i - 1], 3))
        SRer_layers.append(UpScaler(in_channel_list[0], out_channel, 3, is_lastone=True))

        self.SRer = nn.ModuleList(SRer_layers)

    def forward(self, inputs):

        if len(inputs) != len(self.SRer):
            raise ValueError("inputs of FusionNet is not same with cfg['multiscalefeature_outchannel']")

        xd = inputs[len(inputs) - 1]
        for i in range(0, len(self.SRer) - 1):
            xs = inputs[len(inputs) - 2 - i]
            xd = self.SRer[i](xd, xs)
        res = self.SRer[-1](xd)
        return res


class DetectNet1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DetectNet1, self).__init__()
        self.conv1 = Conv2d_Bn_Relu(in_channel, in_channel, 1, 1)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 1, 1)

    def forward(self, feature):
        f = self.conv1(feature)
        result = F.sigmoid(self.conv2(f))
        return result


class DetectNet2(nn.Module):
    def __init__(self, in_channel, out_channel, thre_min=0.01, thre_max=0.5):
        super(DetectNet2, self).__init__()
        self.thre_min = thre_min
        self.thre_max = thre_max
        self.conv = Conv2d_Bn_Relu(in_channel, out_channel, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input, oriP):
        generP = self.conv(input)
        # target = (torch.clamp(oriP - generP, self.thre_min, self.thre_max) - self.thre_min) / (self.thre_max - self.thre_min)
        target = oriP - generP
        return generP, target


class BaseNet1(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(BaseNet1, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler1 = nn.MaxPool2d(3, 2, 1)
        self.resconv = Resconv(in_channel, cfg["resconv_outchannel"], downsampler1)

        downsampler2 = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"], cfg["multiscalefeature_outchannel"], downsampler2
        )
        self.ffusion = FusionNet_plus(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"], cfg["featurefusion_outchannel"]
        )
        self.detect = DetectNet2(cfg["featurefusion_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)
        x11, x21, x31 = self.multiscalef(x)
        xf1 = self.ffusion(x, x11, x21, x31)  # (B, C, 128, 128)
        result = self.detect(xf1)
        return result


class BaseNet2(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(BaseNet2, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler1 = nn.MaxPool2d(3, 2, 1)
        self.resconv = Resconv(in_channel, cfg["resconv_outchannel"], downsampler1)

        downsampler2 = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"], cfg["multiscalefeature_outchannel"], downsampler2
        )
        self.ffusion = FusionNet_plus(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"], cfg["featurefusion_outchannel"]
        )
        self.detect = DetectNet2(cfg["featurefusion_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)
        x11, x21, x31 = self.multiscalef(x)
        xf1 = self.ffusion(x, x11, x21, x31)  # (B, C, 128, 128)
        result = self.detect(xf1)
        return result


class BaseNet3(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(BaseNet3, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler1 = nn.MaxPool2d(3, 2, 1)
        self.resconv = Resconv(in_channel, cfg["resconv_outchannel"], downsampler1)

        downsampler2 = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"],
            cfg["multiscalefeature_outchannel"],
            0,
            cfg,
            downsampler2,
        )
        self.ffusion = FusionNet_upscale(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"], cfg["featurefusion_outchannel"]
        )
        self.detect = DetectNet1(cfg["featurefusion_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)
        outputs_f = self.multiscalef(x)
        xf1 = self.ffusion([x] + outputs_f)  # (B, 32, 256, 256)
        # target = self.detect(xf1, img)
        target = self.detect(xf1)
        return target


class BaseNet4(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(BaseNet4, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        self.resconv = ShallowFeatureExtractor(in_channel, cfg["resconv_outchannel"])    # 256

        self.ds1 = nn.Sequential(ResBlock(cfg["resconv_outchannel"], 32, 3, 1),
                                 ShallowFeatureExtractor(32, 32, 3, 1, Conv2d_Bn_Relu(32, 32, 3, 2, 1)))    # 128
        self.ds2 = nn.Sequential(ResBlock(32, 64, 3, 1), 
                                 ShallowFeatureExtractor(64,   64, 3, 1, Conv2d_Bn_Relu(64, 64, 3, 2, 1)))    # 64
        downsampler = nn.MaxPool2d(3, 2, 1)
        self.ds3 = nn.Sequential(ResBlock(64, 128, 3, 1), 
                                 ShallowFeatureExtractor(128, 128, 3, 1, downsampler))    # 32
        self.ds4 = nn.Sequential(ResBlock(128, 256, 3, 1), 
                                 ShallowFeatureExtractor(256, 256, 3, 1, downsampler)) # 16
        # self.ds5 = nn.Sequential(ResBlock(256, 512, 3, 1), 
        #                          ShallowFeatureExtractor(512, 512, 3, 1, downsampler)) # 8
        # self.ds6 = nn.Sequential(ResBlock(512, 1024, 3, 1), 
        #                          ShallowFeatureExtractor(1024, 1024, 3, 1, downsampler)) # 4
        # self.ds7 = nn.Sequential(ResBlock(256, 256, 3, 1), 
        #                          ShallowFeatureExtractor(256, 256, 3, 1, downsampler)) # 2
        
        # self.us7 = UpScaler_cat(256, 256)   # 4
        # self.us6 = UpScaler_cat(1024, 512)   # 8
        # self.us5 = UpScaler_cat(512, 256)   # 16
        self.us4 = UpScaler_compare(256, 128)   # 32
        self.us3 = UpScaler_compare(128, 64)    # 64
        self.us2 = UpScaler_compare(64, 32)     # 128
        self.us1 = UpScaler_compare(32, cfg["resconv_outchannel"])  # 256

        self.detect = DetectNet1(cfg["resconv_outchannel"], 1)

    def forward(self, img):
        f_256 = self.resconv(img)
        f_128 = self.ds1(f_256)
        f_64 = self.ds2(f_128)
        f_32 = self.ds3(f_64)
        f_16 = self.ds4(f_32)
        # f_8 = self.ds5(f_16)
        # f_4 = self.ds6(f_8)
        # f_2 = self.ds7(f_4)

        # seg_4 = self.us7(f_2, f_4)
        # seg_8 = self.us6(f_4, f_8)
        # seg_16 = self.us5(seg_8, f_16)
        seg_32 = self.us4(f_16, f_32)
        seg_64 = self.us3(seg_32, f_64)
        seg_128 = self.us2(seg_64, f_128)
        seg_256 = self.us1(seg_128, f_256)
        # seg_256 = F.interpolate(seg_128, scale_factor=2, mode='bilinear')

        res = self.detect(seg_256)
        return res


class interact(nn.Module):
    def __init__(self, dist):
        super(interact, self).__init__()
        self.dist = dist

    def forward(self, feature):
        avg_pooled_feature = F.interpolate(F.avg_pool2d(feature, 3, 2, 1), scale_factor=2, mode='nearest')
        top_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=self.dist*2, dims=-2)).pow(2), dim=1, keepdim=True)
        top_right_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=[self.dist*2, -self.dist*2], dims=[-2, -1])).pow(2), dim=1, keepdim=True)
        top_left_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=[self.dist*2, self.dist*2], dims=[-2, -1])).pow(2), dim=1, keepdim=True)
        bot_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=-self.dist*2, dims=-2)).pow(2), dim=1, keepdim=True)
        bot_right_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=[-self.dist*2, -self.dist*2], dims=[-2, -1])).pow(2), dim=1, keepdim=True)
        bot_left_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=[-self.dist*2, self.dist*2], dims=[-2, -1])).pow(2), dim=1, keepdim=True)
        right_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=-self.dist*2, dims=-1)).pow(2), dim=1, keepdim=True)
        left_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=self.dist*2, dims=-1)).pow(2), dim=1, keepdim=True)
        diff = torch.concatenate((top_left_f, top_f, top_right_f, left_f, right_f, bot_left_f, bot_f, bot_right_f), dim=1)
        diff = torch.max(diff, dim=1, keepdim=True).values
        return diff
        

class BaseNet5(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(BaseNet5, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        self.resconv = ShallowFeatureExtractor(in_channel, cfg["resconv_outchannel"])    # 256

        self.ds1 = nn.Sequential(ResBlock(cfg["resconv_outchannel"], 32, 3, 1),
                                 ShallowFeatureExtractor(32, 32, 3, 1, Conv2d_Bn_Relu(32, 32, 3, 2, 1)))    # 128
        self.ds2 = nn.Sequential(ResBlock(32, 64, 3, 1), 
                                 ShallowFeatureExtractor(64,   64, 3, 1, Conv2d_Bn_Relu(64, 64, 3, 2, 1)))    # 64
        self.ds3 = nn.Sequential(ResBlock(64, 128, 3, 1), 
                                 ShallowFeatureExtractor(128, 128, 3, 1, Conv2d_Bn_Relu(128, 128, 3, 2, 1)))    # 32

        self.contrast_1 = interact(1)   #(B, 1, 32, 32)
        self.contrast_3 = interact(3)
        self.contrast_5 = interact(5)
        
        self.us3 = UpScaler_compare(128, 64)    # 64
        self.us2 = UpScaler_compare(64, 32)     # 128
        self.us1 = UpScaler_compare(32, cfg["resconv_outchannel"])  # 256

        self.detect = DetectNet1(cfg["resconv_outchannel"], 1)

    def forward(self, img):
        f_256 = self.resconv(img)
        f_128 = self.ds1(f_256)
        f_64 = self.ds2(f_128)
        f_32 = self.ds3(f_64)

        contrast1 = self.contrast_1(f_32)
        contrast2 = self.contrast_3(f_32)
        contrast3 = self.contrast_5(f_32)

        f_32 = f_32 * (contrast1 + contrast2 + contrast3)
        
        seg_64 = self.us3(f_32, f_64)
        seg_128 = self.us2(seg_64, f_128)
        seg_256 = self.us1(seg_128, f_256)
        # seg_256 = F.interpolate(seg_128, scale_factor=2, mode='bilinear')

        res = self.detect(seg_256)
        return res


class BaseNetWithLoss(nn.Module):
    def __init__(self, cfg, net, loss_fn):
        super(BaseNetWithLoss, self).__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.cfg = cfg

    def forward(self, img, label, current_loss=1.0):
        res = self.net(img)
        loss = self.loss_fn(res, label)
        return res, loss


class LargeBaseNet(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(LargeBaseNet, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler = nn.MaxPool2d(2, 2)
        self.resconv = Resconv(in_channel, cfg["resconv_outchannel"])

        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"], cfg["multiscalefeature_outchannel"], downsampler
        )
        self.ffusion = FeatureFusionNet2(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"],
            cfg["featurefusion_outchannel"],
            cfg["ratio_list"],
        )
        self.detect = DetectNet4(cfg["featurefusion_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)  # (B, C, 256, 256)
        x11, x21, x31 = self.multiscalef(x)
        xf1 = self.ffusion(x, x11, x21, x31)  # (B, C, 256, 256)
        result = self.detect(xf1)
        return result


class LargeBaseNet2(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(LargeBaseNet2, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler1 = nn.MaxPool2d(3, 2, 1)
        self.resconv_deep = Resconv(in_channel, cfg["resconv_outchannel"], downsampler1)
        self.resconv_shallow = Resconv(in_channel, cfg["resconv_outchannel"])

        downsampler2 = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"], cfg["multiscalefeature_outchannel"], downsampler2
        )
        self.ffusion = FeatureFusionNet2(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"],
            cfg["featurefusion_outchannel"],
            cfg["ratio_list"],
        )
        self.detect = DetectNet4(cfg["featurefusion_outchannel"] + cfg["resconv_outchannel"], 1)

    def forward(self, img):
        x1 = self.resconv_deep(img)  # (B, C, 128, 128)
        x11, x21, x31 = self.multiscalef(x1)
        xf1 = self.ffusion(x1, x11, x21, x31)  # (B, C, 128, 128)
        x2 = self.resconv_shallow(img)
        xf1 = torch.cat((x2, xf1.repeat(1, 1, 2, 2)), dim=1)
        result = self.detect(xf1)
        return result


class GaussNet(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(GaussNet, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler1 = nn.MaxPool2d(3, 2, 1)
        self.resconv = Resconv(in_channel, cfg["resconv_outchannel"], downsampler1)

        downsampler2 = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"],
            cfg["multiscalefeature_outchannel"],
            1,
            cfg,
            downsampler2,
        )
        self.ffusion = FusionNet_upscale(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"], cfg["featurefusion_outchannel"]
        )
        # self.detect = DetectNet2(
        #     cfg["featurefusion_outchannel"], 1, cfg["noise_filter_threshold"], cfg["background_aug_threshold"]
        # )
        self.detect = DetectNet1(cfg["featurefusion_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)
        outputs_f = self.multiscalef(x)
        xf1 = self.ffusion([x] + outputs_f)  # (B, 32, 256, 256)
        # target = self.detect(xf1, img)
        target = self.detect(xf1)
        return target


class GaussNet2(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(GaussNet2, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler1 = nn.MaxPool2d(3, 2, 1)
        self.resconv = Gauss_Resconv(in_channel, cfg["resconv_outchannel"], cfg["gauss_sigma"], downsampler1)

        downsampler2 = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"],
            cfg["multiscalefeature_outchannel"],
            0,
            cfg,
            downsampler2,
        )
        self.ffusion = FusionNet_upscale(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"], cfg["featurefusion_outchannel"]
        )
        self.detect = DetectNet1(cfg["featurefusion_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)
        outputs_f = self.multiscalef(x)
        xf1 = self.ffusion([x] + outputs_f)  # (B, 32, 256, 256)
        # target = self.detect(xf1, img)
        target = self.detect(xf1)
        return target


class GaussNet3(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(GaussNet3, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler1 = nn.MaxPool2d(3, 2, 1)
        self.resconv = Resconv(in_channel, cfg["resconv_outchannel"], downsampler1)

        downsampler2 = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"],
            cfg["multiscalefeature_outchannel"],
            1,
            cfg,
            downsampler2,
        )
        self.ffusion = FusionNet_upscale(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"], cfg["featurefusion_outchannel"]
        )
        self.detect = DetectNet1(cfg["featurefusion_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)
        outputs_f = self.multiscalef(x)
        xf1 = self.ffusion([x] + outputs_f)  # (B, 32, 256, 256)
        target = self.detect(xf1)
        return target


class GaussNet4(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(GaussNet4, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler1 = nn.MaxPool2d(3, 2, 1)
        self.resconv = Gauss_Resconv(
            in_channel, cfg["resconv_outchannel"], cfg["gauss_kernelsize"], cfg["gauss_sigma"], downsampler1
        )

        downsampler2 = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"],
            cfg["multiscalefeature_outchannel"],
            1,
            cfg,
            downsampler2,
        )
        self.ffusion = FusionNet_upscale(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"], cfg["featurefusion_outchannel"]
        )
        self.detect = DetectNet1(cfg["featurefusion_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)
        outputs_f = self.multiscalef(x)
        xf1 = self.ffusion([x] + outputs_f)  # (B, 32, 256, 256)
        target = self.detect(xf1)
        return target


class GaussNet5(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
         
        self.ffusion = FusionNet_upscale(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"], cfg["featurefusion_outchannel"]
        )
        self.detect = DetectNet1(cfg["featurefusion_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)
        outputs_f = self.multiscalef(x)
        xf1 = self.ffusion([x] + outputs_f)  # (B, 32, 256, 256)
        target = self.detect(xf1)
        return target


class SigmoidNet(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(SigmoidNet, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler1 = nn.MaxPool2d(3, 2, 1)
        self.resconv = Resconv(in_channel, cfg["resconv_outchannel"], downsampler1)

        downsampler2 = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"],
            cfg["multiscalefeature_outchannel"],
            2,
            cfg,
            downsampler2,
        )
        self.ffusion = FusionNet_upscale(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"], cfg["featurefusion_outchannel"]
        )
        self.detect = DetectNet1(cfg["featurefusion_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)
        outputs_f = self.multiscalef(x)
        xf1 = self.ffusion([x] + outputs_f)  # (B, 32, 256, 256)
        # target = self.detect(xf1, img)
        target = self.detect(xf1)
        return target


class YoloNet(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(YoloNet, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler = nn.MaxPool2d(2, 2)
        self.resconv = Resconv(in_channel, cfg["resconv_outchannel"], downsampler)

        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"], cfg["multiscalefeature_outchannel"], downsampler
        )
        self.ffusion = FeatureFusionNet2(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"],
            cfg["featurefusion_outchannel"],
            cfg["ratio_list"],
        )
        self.detect = DetectNet4(cfg["featurefusion_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)  # (B, C, 256, 256)
        x11, x21, x31 = self.multiscalef(x)
        xf1 = self.ffusion(x, x11, x21, x31)  # (B, C, 256, 256)
        result = self.detect(xf1)
        return result
