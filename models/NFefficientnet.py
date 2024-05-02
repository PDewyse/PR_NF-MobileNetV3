# model code based on https://github.com/EstherBear/small-net-cifar100
import torch.nn as nn
import math
# from tools.utils import conv_bn_act
# from tools.utils import SameConv
# from tools.utils import SE
# from tools.utils import drop_connect
# from tools.utils import swish

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class Swish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.mul_(torch.sigmoid(x))
        else:
            return x * torch.sigmoid(x, inplace=self.inplace)
    
# Add variance preserving swish function
class VPSwish(nn.Module):
    """Variance preserving swish activation function"""
    def __init__(self):
        super().__init__()
        self.activation = Swish()

    def forward(self, x):
        return self.activation(x) * 1.787301

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
class conv_NF_act(nn.Module):
    def __init__(self, inchannels, outchannels, kernelsize, stride=1, dilation=1, groups=1, bias=False, bn_momentum=0.99):
        super().__init__()
        self.block = nn.Sequential(
            WSSameConv(inchannels, outchannels, kernelsize, stride, dilation, groups, bias=bias),
            # nn.BatchNorm2d(outchannels, momentum=1-bn_momentum),
            VPSwish()
        )

    def forward(self, x):
        return self.block(x)

# Added: made it so this one also implements WSConv2D
class WSSameConv(nn.Conv2d):
    # coming from WSConv2D
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0,
        dilation = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
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
    #

    def how_padding(self, n, kernel, stride, dilation):
        out_size = (n + stride - 1) // stride
        real_kernel = (kernel - 1) * dilation + 1
        padding_needed = max(0, (out_size - 1) * stride + real_kernel - n)
        is_odd = padding_needed % 2
        return padding_needed, is_odd

    def forward(self, x):
        row_padding_needed, row_is_odd = self.how_padding(x.size(2), self.weight.size(2), self.stride[0], self.dilation[0])
        col_padding_needed, col_is_odd = self.how_padding(x.size(3), self.weight.size(3), self.stride[1], self.dilation[1])
        if row_is_odd or col_is_odd:
            x = F.pad(x, [0, col_is_odd, 0, row_is_odd])
        return F.conv2d(input = x, 
                        weight = self.standardized_weights(), # change for weight scaling here
                        bias =self.bias, 
                        stride =self.stride,
                        padding = (row_padding_needed//2, col_padding_needed//2),
                        dilation = self.dilation,
                        groups = self.groups)
                        
    #def count_your_model(self, x, y):
     #   return y.size(2) * y.size(3) * y.size(1) * self.weight.size(2) * self.weight.size(3) / self.groups
class SE(nn.Module):
    def __init__(self, inchannels, mid):
        super().__init__()
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(inchannels, mid),
            VPSwish(),
            nn.Linear(mid, inchannels)
        )

    def forward(self, x):
        out = self.AvgPool(x)
        out = out.view(x.size(0), -1)
        out = self.SEblock(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return x * torch.sigmoid(out) * 2 # as in the paper, see B.3 under squeeze and excitation layer


class drop_connect(nn.Module):
    def __init__(self, survival=0.8):
        super().__init__()
        self.survival = survival

    def forward(self, x):
        if not self.training:
            return x

        random = torch.rand((x.size(0), 1, 1, 1), device=x.device) # 涉及到x的属性的步骤，直接挪到forward
        random += self.survival
        random.requires_grad = False
        return x / self.survival * torch.floor(random)

class MBConv(nn.Module):
    def __init__(self, inchannels, outchannels, expan, kernelsize, stride, se_ratio=4,
                 is_skip=True, dc_ratio=(1-0.8), bn_momentum=0.90, beta:float=1.0, alpha:float=0.2):
        super().__init__()
        mid = expan * inchannels
        self.pointwise1 = conv_NF_act(inchannels, mid, 1) if expan != 1 else nn.Identity()
        self.depthwise = conv_NF_act(mid, mid, kernelsize, stride=stride, groups=mid)
        self.se = SE(mid, int(inchannels/se_ratio))
        self.pointwise2 = nn.Sequential(
            WSSameConv(mid, outchannels, 1),
            # nn.BatchNorm2d(outchannels, 1-bn_momentum)
        )
        self.skip = is_skip and inchannels == outchannels and stride == 1
        # self.dc = drop_connect(1-dc_ratio)
        self.dc = nn.Identity()
        # added
        self.activation = VPSwish()
        self.beta, self.alpha = beta, alpha
        #

    def forward(self, x):
        # added: Downscale the input to each residual branch
        x = self.activation(x) * self.beta
        #
        residual = self.pointwise1(x)
        residual = self.depthwise(residual)
        residual = self.se(residual)
        residual = self.pointwise2(residual)
        # added: downscale the input to the convolution on the skip path in transition blocks
        residual = residual * self.alpha
        #
        if self.skip:
            residual = self.dc(residual)
            out = residual + x
        else:
            out = residual

        return out


class MBblock(nn.Module):
    def __init__(self, inchannels, outchannels, expan, kernelsize, stride, se_ratio, repeat,
                 is_skip, dc_ratio=(1-0.8), bn_momentum=0.90, beta:float=1.0, alpha:float=0.2):
        super().__init__()

        layers = []
        layers.append(MBConv(inchannels, outchannels, expan, kernelsize, stride,
                             se_ratio, is_skip, dc_ratio, bn_momentum, beta, alpha))
        while repeat-1:
            layers.append(MBConv(outchannels, outchannels, expan, kernelsize, 1,
                                 se_ratio, is_skip, dc_ratio, bn_momentum, beta, alpha))
            repeat = repeat - 1

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # for downscaling see MBConv
        return self.block(x)


class NFEfficientNet(nn.Module):
    def __init__(self, width_multipler, depth_multipler, do_ratio, min_width=0, width_divisor=8,
                 se_ratio=4, dc_ratio=(1-0.8), bn_momentum=0.90, num_class=100, alpha:float=0.2):
        super().__init__()

        def renew_width(x):
            min = max(min_width, width_divisor)
            x *= width_multipler
            new_x = max(min, int((x + width_divisor/2) // width_divisor * width_divisor))

            if new_x < 0.9 * x:
                new_x += width_divisor
            return int(new_x)

        def renew_depth(x):
            return int(math.ceil(x * depth_multipler))

        self.stage1 = nn.Sequential(
            WSSameConv(3, renew_width(32), 3),
            # nn.BatchNorm2d(renew_width(32), momentum=bn_momentum),
            VPSwish()
        )
        # added: Compute and forward propagate the expected signal variance
        # we need to pass a beta which is calculated via beta = 1 / expected_std, this std changes with alfa (scaler hyperparameter)
        # and we just need to pass the changing beta that we can calculate from the expected_std and alfa per residual block (??!)
        expected_std = 1.0
        betas = []
        for _ in range(7):
            betas.append(1.0/expected_std)
            expected_std = (expected_std **2 + alpha**2)**0.5
        
        self.stage2 = nn.Sequential(
                    # inchannels     outchannels  expand k  s(mobilenetv2)  repeat      is_skip
            MBblock(renew_width(32), renew_width(16), 1, 3, 1, se_ratio, renew_depth(1), True, dc_ratio, bn_momentum, beta=betas[0], alpha=alpha),
            MBblock(renew_width(16), renew_width(24), 6, 3, 2, se_ratio, renew_depth(2), True, dc_ratio, bn_momentum, beta=betas[1], alpha=alpha),
            MBblock(renew_width(24), renew_width(40), 6, 5, 2, se_ratio, renew_depth(2), True, dc_ratio, bn_momentum, beta=betas[2], alpha=alpha),
            MBblock(renew_width(40), renew_width(80), 6, 3, 2, se_ratio, renew_depth(3), True, dc_ratio, bn_momentum, beta=betas[3], alpha=alpha),
            MBblock(renew_width(80), renew_width(112), 6, 5, 1, se_ratio, renew_depth(3), True, dc_ratio, bn_momentum, beta=betas[4], alpha=alpha),
            MBblock(renew_width(112), renew_width(192), 6, 5, 1, se_ratio, renew_depth(4), True, dc_ratio, bn_momentum, beta=betas[5], alpha=alpha),
            MBblock(renew_width(192), renew_width(320), 6, 3, 1, se_ratio, renew_depth(1), True, dc_ratio, bn_momentum, beta=betas[6], alpha=alpha)
        )
        #print("initing stage 3")
        self.stage3 = nn.Sequential(
            WSSameConv(renew_width(320), renew_width(1280), 1, stride=1),
            # nn.BatchNorm2d(renew_width(1280), bn_momentum),
            VPSwish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(do_ratio)
        )
        self.FC = nn.Linear(renew_width(1280), num_class)
        #print("initing weights")

        self.init_weights()
        #print("finish initing")

    def init_weights(self):
        # SameConv用kaiming, Linear用1/sqrt(channels)
        for m in self.modules():
            if isinstance(m, WSSameConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                bound = 1/int(math.sqrt(m.weight.size(1)))
                nn.init.uniform(m.weight, -bound, bound)

    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = out.view(out.size(0), -1)
        out = self.FC(out)
        return out


def NFefficientnet(width_multipler, depth_multipler, num_class=100, bn_momentum=0.90, do_ratio=0.2):
    return NFEfficientNet(width_multipler, depth_multipler,
                        num_class=num_class, bn_momentum=bn_momentum, do_ratio=do_ratio)




