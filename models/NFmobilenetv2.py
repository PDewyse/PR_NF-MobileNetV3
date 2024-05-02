# model code based on https://github.com/EstherBear/small-net-cifar100
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add variance preserving relu6 function
class VPReLU6(nn.Module):
    """Variance preserving ReLU6 activation function"""
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x, inplace=self.inplace)* 1.7133749

# added as alternative to conv2D for designing NF-MobileNetV3
class WSConv2D(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0,
        dilation = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):

        super(WSConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
        nn.init.xavier_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer('eps', torch.tensor(1e-4, requires_grad=False), persistent=False)
        self.register_buffer('fan_in', torch.tensor(np.prod(self.weight.shape[1:]), requires_grad=False).type_as(self.weight), persistent=False)

    def standardized_weights(self):
        # Original code: HWCN
        mean = torch.mean(self.weight, axis=[1,2,3], keepdims=True)
        var = torch.var(self.weight, axis=[1,2,3], keepdims=True)
        scale = torch.rsqrt(torch.maximum(var * self.fan_in, self.eps))
        return (self.weight - mean) * scale * self.gain
        
    def forward(self, x):
        return F.conv2d(
            input=x,
            weight=self.standardized_weights(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

class bottleneck(nn.Module):
    def __init__(self, inchannels, outchannels, stride, expansion, beta:float=1.0, VPalpha:float=0.2):
        super().__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.stride = stride
        # added
        self.beta, self.VPalpha = beta, VPalpha
        self.activation = VPReLU6(inplace=True)
        #

        self.residual = nn.Sequential(
            WSConv2D(in_channels=inchannels, out_channels=expansion*inchannels, kernel_size=1),
            # nn.BatchNorm2d(expansion*inchannels),
            VPReLU6(inplace=True),

            WSConv2D(in_channels=inchannels*expansion, out_channels=inchannels*expansion, kernel_size=3, padding=1,
                      groups=inchannels*expansion, stride=stride),
            # nn.BatchNorm2d(expansion*inchannels),
            VPReLU6(inplace=True),

            WSConv2D(in_channels=expansion * inchannels, out_channels=outchannels, kernel_size=1),
            # nn.BatchNorm2d(outchannels)
        )

    def forward(self, x):
        # added: Downscale the input to each residual branch
        x = self.activation(x) * self.beta
        #
        out = self.residual(x)
        # added: downscale the input to the convolution on the skip path in transition blocks
        out = out * self.VPalpha
        #
        if self.inchannels == self.outchannels and self.stride == 1:
            out += x
        return out


class NFMobileNetV2(nn.Module):
    def __init__(self, alpha=1, num_class=100, VPalpha:float=0.2):
        super().__init__()

        self.Conv1 = nn.Sequential(
            WSConv2D(in_channels=3, out_channels=int(alpha*32), kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            VPReLU6(inplace=True)
        )
        # added: Compute and forward propagate the expected signal variance
        # we need to pass a beta which is calculated via beta = 1 / expected_std, this std changes with alfa (scaler hyperparameter)
        # and we just need to pass the changing beta that we can calculate from the expected_std and alfa per residual block (??!)
        expected_std = 1.0
        betas = []
        for _ in range(7):
            betas.append(1.0/expected_std)
            expected_std = (expected_std **2 + alpha**2)**0.5

        self.stage1 = bottleneck(int(alpha*32), 16, 1, 1, beta=betas[0], VPalpha=VPalpha)
        self.stage2 = self.make_layer(int(alpha*16), 6, int(alpha*24), 2, 2, beta=betas[1], VPalpha=VPalpha)
        self.stage3 = self.make_layer(int(alpha*24), 6, int(alpha*32), 3, 2, beta=betas[2], VPalpha=VPalpha)
        self.stage4 = self.make_layer(int(alpha*32), 6, int(alpha*64), 4, 2, beta=betas[3], VPalpha=VPalpha)
        self.stage5 = self.make_layer(int(alpha*64), 6, int(alpha*96), 3, 1, beta=betas[4], VPalpha=VPalpha)
        self.stage6 = self.make_layer(int(alpha*96), 6, int(alpha*160), 3, 1, beta=betas[5], VPalpha=VPalpha)
        self.stage7 = self.make_layer(int(alpha*160), 6, int(alpha*320), 1, 1, beta=betas[6], VPalpha=VPalpha)

        self.Conv2 = nn.Sequential(
            WSConv2D(in_channels=int(alpha*320), out_channels=1280, kernel_size=1),
            # nn.BatchNorm2d(1280),
            VPReLU6(inplace=True)
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.Conv3 = WSConv2D(in_channels=1280, out_channels=num_class, kernel_size=1)

    def make_layer(self, inchannels, t, outchannels, n, s, beta:float=1.0, VPalpha:float=0.2):
        layer = []
        layer.append(bottleneck(inchannels, outchannels, s, t, beta=beta, VPalpha=VPalpha))
        n = n - 1
        while n:
            layer.append(bottleneck(outchannels, outchannels, 1, t, beta=beta, VPalpha=VPalpha))
            n -= 1
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.stage6(out)
        out = self.stage7(out)
        out = self.Conv2(out)
        out = self.AvgPool(out)
        out = self.drop(out)
        out = self.Conv3(out)
        out = out.view(out.size(0), -1)

        return out


def NFmobilenetv2(alpha=1, num_class=100):
    return NFMobileNetV2(alpha, num_class)