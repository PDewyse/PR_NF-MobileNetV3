# model code based on https://github.com/EstherBear/small-net-cifar100
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# add the variance preserving activation functions using the nonlinearity-specific constant gamma
class VPReLU(nn.Module):
    def __init__(self, inplace: bool = True):
        super(VPReLU, self).__init__()
        self.inplace = inplace
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input, inplace=True) * 1.7139588594436646
# 
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
#
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inchannels, outchannels, stride, **kwargs):
        super().__init__()
        self.DepthwiseConv = nn.Sequential(
            WSConv2D(in_channels=inchannels, out_channels=inchannels, kernel_size=3,
                      padding=1, stride=stride, groups=inchannels, bias=False, **kwargs),
            # nn.BatchNorm2d(inchannels),
            VPReLU(inplace=True)

        )
        self.PointwiseConv = nn.Sequential(
            WSConv2D(in_channels=inchannels, out_channels=outchannels, kernel_size=1, bias=False, stride=1),
            # nn.BatchNorm2d(outchannels),
            VPReLU(inplace=True)
        )

    def forward(self, x):
        out = self.DepthwiseConv(x)
        out = self.PointwiseConv(out)
        return out

class FullConv(nn.Module):
    def __init__(self, inchannels, outchannels, stride, **kwargs):
        super().__init__()
        self.Conv = nn.Sequential(
            WSConv2D(in_channels=inchannels, out_channels=outchannels, bias=False, kernel_size=3,
                      padding=1, stride=stride, **kwargs),
            # nn.BatchNorm2d(outchannels),
            VPReLU(inplace=True)
        )

    def forward(self, x):
        out = self.Conv(x)
        return out

class NFMobileNet32(nn.Module):
    def __init__(self, alpha=1, num_calss=100):
        super().__init__()
        self.alpha = alpha
        self.num_class = num_calss
        # Conv separated by down sampling
        self.Conv1 = nn.Sequential(
            FullConv(inchannels=3, outchannels=int(alpha*32), stride=1),
            DepthwiseSeparableConv(inchannels=int(alpha*32), outchannels=int(alpha*64), stride=1)
        )
        self.Conv2 = nn.Sequential(
            DepthwiseSeparableConv(inchannels=int(alpha*64), outchannels=int(alpha*128), stride=2),
            DepthwiseSeparableConv(inchannels=int(alpha*128), outchannels=int(alpha*128), stride=1)
        )
        self.Conv3 = nn.Sequential(
            DepthwiseSeparableConv(inchannels=int(alpha*128), outchannels=int(alpha*256), stride=2),
            DepthwiseSeparableConv(inchannels=int(alpha*256), outchannels=int(alpha*256), stride=1)
        )

        self.Conv4 = nn.Sequential(
            DepthwiseSeparableConv(inchannels=int(alpha*256), outchannels=int(alpha*512), stride=2),
            DepthwiseSeparableConv(inchannels=int(alpha*512), outchannels=int(alpha*512), stride=1),
            DepthwiseSeparableConv(inchannels=int(alpha*512), outchannels=int(alpha*512), stride=1),
            DepthwiseSeparableConv(inchannels=int(alpha*512), outchannels=int(alpha*512), stride=1),
            DepthwiseSeparableConv(inchannels=int(alpha*512), outchannels=int(alpha*512), stride=1),
            DepthwiseSeparableConv(inchannels=int(alpha*512), outchannels=int(alpha*512), stride=1)
        )

        self.Conv5 = nn.Sequential(
            DepthwiseSeparableConv(inchannels=int(alpha*512), outchannels=int(alpha*1024), stride=2),
            DepthwiseSeparableConv(inchannels=int(alpha*1024), outchannels=int(alpha*1024), stride=1)
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.FC = nn.Linear(int(alpha*1024), num_calss)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Conv2(out)
        out = self.Conv3(out)
        out = self.Conv4(out)
        out = self.Conv5(out)
        out = self.AvgPool(out)
        out = out.view(x.size(0), -1)
        out = self.drop(out)
        out = self.FC(out)
        return out


def NFmobilenet(alpha=1, num_class=100):
    return NFMobileNet32(alpha, num_class)