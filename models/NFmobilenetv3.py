# model code based on https://github.com/akrapukhin/MobileNetV3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Hsigmoid(nn.Module):
    """
    Hard sigmoid function
    """
    def __init__(self, inplace: bool = True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu6(input + 3.0, inplace=self.inplace) * (1.0/6.0)
    
class VPHsigmoid(nn.Module):
    """
    Variance preserving Hard sigmoid function
    """
    def __init__(self, inplace: bool = True):
        super(VPHsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu6(input + 3.0, inplace=self.inplace) * (1.0/6.0) * 6.018489
    
class Hswish(nn.Module):
    """
    Hard swish function
    """
    def __init__(self, inplace: bool = True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu6(input+3.0, inplace=self.inplace) * (1.0/6.0) * input

class VPHswish(nn.Module):
    """
    Variance preserving Hard swish function
    """
    def __init__(self, inplace: bool = True):
        super(VPHswish, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu6(input+3.0, inplace=self.inplace) * (1.0/6.0) * input * 1.8138213

class VPReLU(nn.Module):
    def __init__(self, inplace: bool = True):
        super(VPReLU, self).__init__()
        self.inplace = inplace
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input, inplace=True) * 1.7139588594436646
    
class WSConv2D(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0,
        dilation = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):

        super(WSConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
        nn.init.xavier_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer('eps', torch.tensor(1e-4, requires_grad=False), persistent=False)
        self.register_buffer('fan_in', torch.tensor(np.prod(self.weight.shape[1:]), requires_grad=False).type_as(self.weight), persistent=False)

    def standardized_weights(self):
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

class Squeeze_excite(nn.Module):
    def __init__(self, num_channels, r=4):
        """
        Squeeze-and-Excitation block
          Args:
            num_channels (int): number of channels in the input tensor
            r (int): num_channels are divided by r in the first conv block
        """
        super(Squeeze_excite, self).__init__()
        self.conv_0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            WSConv2D(num_channels, num_channels//r, kernel_size=1),
            VPReLU(inplace=True)
        )
        self.conv_1 = nn.Sequential(
            WSConv2D(num_channels//r, num_channels, kernel_size=1),
            VPHsigmoid()
        )
 
    def forward(self, input):
        out = self.conv_0(input)
        out = self.conv_1(out)
        out = out * input
        return out

class Bneck(nn.Module):
    def __init__(self, nin, nout, k_size, exp_size, se, act, s, wm=1.0, beta:float=1.0, alpha:float=0.2):
        """
        Bottleneck block
          Args:
            nin (int): number of channels in the input tensor
            nout (int): number of channels in the output tensor
            k_size (int): size of filters
            exp_size (int): expansion size
            se (bool): whether to use Squeeze-and-Excitation
            act (nn.Module): activation function
            s (int): stride
            wm (float): width multiplier
        """
        super(Bneck, self).__init__()
        nin = int(nin*wm)
        nout = int(nout*wm)
        exp_size = int(exp_size*wm)
        self.activation = act()
        self.beta, self.alpha = beta, alpha
        self.pointwise_0 = nn.Sequential(
            WSConv2D(nin, exp_size, kernel_size=1, bias=False),
            act(inplace=True)
        )

        self.depthwise_1 = nn.Sequential(
            WSConv2D(exp_size, exp_size, kernel_size=k_size, padding=(k_size-1)//2, groups=exp_size, stride=s, bias=False),
            act(inplace=True)
        )
        self.se = se
        self.se_block = Squeeze_excite(num_channels=exp_size, r=4)
        self.pointwise_2 = nn.Sequential(
            WSConv2D(exp_size, nout, kernel_size=1, bias=False),
        ) 
        self.shortcut = s == 1 and nin == nout

    def forward(self, input):
        input = self.activation(input) * self.beta
        identity = input
        out = self.pointwise_0(input)
        out = self.depthwise_1(out)
        
        if self.se:
            out = self.se_block(out)
        
        out = self.pointwise_2(out)
        out = out * self.alpha
        if self.shortcut:
            out += identity
        return out

class NFMobilenet_v3_large(nn.Module):
    def __init__(self, wm=1.0, si=1, drop_prob=0.0, alpha:float=0.2):
        """
        NFMobilenet v3 large model
          Args:
            wm (float): width multiplier
            si (int): stride in initial layers (set to 1 by default instead of 2 to adapt for small 32x32 resolution of CIFAR)
            drop_prob (float): probability that a neuron is removed for nn.Dropout layer during training
        """
        super(NFMobilenet_v3_large, self).__init__()
        self.wm = wm
        self.conv_0 = nn.Sequential(
            WSConv2D(3, int(16*wm), 3, padding=1, stride=si, bias=False),
            VPHswish()
        )
        expected_std = 1.0
        betas = []
        for _ in range(15):
            betas.append(1.0/expected_std)
            expected_std = (expected_std **2 + alpha**2)**0.5
       
        self.bottlenecks_1 = nn.Sequential(
            Bneck(nin=16,  nout=16,  k_size=3, exp_size=16,  se=False, act=VPReLU, s=1,  wm=wm, beta=betas[0], alpha=alpha),#1
            Bneck(nin=16,  nout=24,  k_size=3, exp_size=64,  se=False, act=VPReLU, s=si, wm=wm, beta=betas[1], alpha=alpha),#2
            Bneck(nin=24,  nout=24,  k_size=3, exp_size=72,  se=False, act=VPReLU, s=1,  wm=wm, beta=betas[2], alpha=alpha),#3
            Bneck(nin=24,  nout=40,  k_size=5, exp_size=72,  se=True,  act=VPReLU, s=si, wm=wm, beta=betas[3], alpha=alpha),#4
            Bneck(nin=40,  nout=40,  k_size=5, exp_size=120, se=True,  act=VPReLU, s=1,  wm=wm, beta=betas[4], alpha=alpha),#5
            Bneck(nin=40,  nout=40,  k_size=5, exp_size=120, se=True,  act=VPReLU, s=1,  wm=wm, beta=betas[5], alpha=alpha),#6
            Bneck(nin=40,  nout=80,  k_size=3, exp_size=240, se=False, act=VPHswish,  s=2,  wm=wm, beta=betas[6], alpha=alpha),#7
            Bneck(nin=80,  nout=80,  k_size=3, exp_size=200, se=False, act=VPHswish,  s=1,  wm=wm, beta=betas[7], alpha=alpha),#8
            Bneck(nin=80,  nout=80,  k_size=3, exp_size=184, se=False, act=VPHswish,  s=1,  wm=wm, beta=betas[8], alpha=alpha),#9
            Bneck(nin=80,  nout=80,  k_size=3, exp_size=184, se=False, act=VPHswish,  s=1,  wm=wm, beta=betas[9], alpha=alpha),#10
            Bneck(nin=80,  nout=112, k_size=3, exp_size=480, se=True,  act=VPHswish,  s=1,  wm=wm, beta=betas[10], alpha=alpha),#11
            Bneck(nin=112, nout=112, k_size=3, exp_size=672, se=True,  act=VPHswish,  s=1,  wm=wm, beta=betas[11], alpha=alpha),#12
            Bneck(nin=112, nout=160, k_size=5, exp_size=672, se=True,  act=VPHswish,  s=2,  wm=wm, beta=betas[12], alpha=alpha),#13
            Bneck(nin=160, nout=160, k_size=5, exp_size=960, se=True,  act=VPHswish,  s=1,  wm=wm, beta=betas[13], alpha=alpha),#14
            Bneck(nin=160, nout=160, k_size=5, exp_size=960, se=True,  act=VPHswish,  s=1,  wm=wm, beta=betas[14], alpha=alpha) #15
        )
        self.conv_2 = nn.Sequential(
            WSConv2D(int(160*wm), int(960*wm), 1, bias=False),
            VPHswish()
        )
        self.conv_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            WSConv2D(int(960*wm), 1280, 1),
            VPHswish(),
            nn.Dropout(p=drop_prob)
        )
        self.conv_4 = WSConv2D(1280, 100, 1)

    def forward(self, input):
        x = self.conv_0(input)
        x = self.bottlenecks_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x.view(x.shape[0], -1)
    
    def name(self):
        return "NFMobilenet_v3_large_" + str(self.wm)

class NFMobilenet_v3_small(nn.Module):
    def __init__(self, wm=1.0, si=1, drop_prob=0.0, alpha:float=0.1):
        """
        NFMobilenet v3 small model
          Args:
            wm (float): width multiplier
            si (int): stride in initial layers (set to 1 by default instead of 2 to adapt for small 32x32 resolution of CIFAR)
            drop_prob (float): probability that a neuron is removed for nn.Dropout layer
        """
        super(NFMobilenet_v3_small, self).__init__()
        self.wm = wm
        self.conv_0 = nn.Sequential(
            WSConv2D(3, int(16*wm), 3, padding=1, stride=si, bias=False),
            VPHswish()
        )
        expected_std = 1.0
        betas = []
        for _ in range(11):
            betas.append(1.0/expected_std)
            expected_std = (expected_std **2 + alpha**2)**0.5

        self.bottlenecks_1 = nn.Sequential(
            Bneck(nin=16,  nout=16,  k_size=3, exp_size=16,  se=True,  act=VPReLU, s=si, wm=wm, beta=betas[0], alpha=alpha),#1 
            Bneck(nin=16,  nout=24,  k_size=3, exp_size=72,  se=False, act=VPReLU, s=si, wm=wm, beta=betas[1], alpha=alpha),#2
            Bneck(nin=24,  nout=24,  k_size=3, exp_size=88,  se=False, act=VPReLU, s=1,  wm=wm, beta=betas[2], alpha=alpha),#3
            Bneck(nin=24,  nout=40,  k_size=5, exp_size=96,  se=True,  act=VPHswish,  s=2,  wm=wm, beta=betas[3], alpha=alpha),#4
            Bneck(nin=40,  nout=40,  k_size=5, exp_size=240, se=True,  act=VPHswish,  s=1,  wm=wm, beta=betas[4], alpha=alpha),#5
            Bneck(nin=40,  nout=40,  k_size=5, exp_size=240, se=True,  act=VPHswish,  s=1,  wm=wm, beta=betas[5], alpha=alpha),#6
            Bneck(nin=40,  nout=48,  k_size=5, exp_size=120, se=True,  act=VPHswish,  s=1,  wm=wm, beta=betas[6], alpha=alpha),#7
            Bneck(nin=48,  nout=48,  k_size=5, exp_size=144, se=True,  act=VPHswish,  s=1,  wm=wm, beta=betas[7], alpha=alpha),#8
            Bneck(nin=48,  nout=96,  k_size=5, exp_size=288, se=True,  act=VPHswish,  s=2,  wm=wm, beta=betas[8], alpha=alpha),#9
            Bneck(nin=96,  nout=96,  k_size=5, exp_size=576, se=True,  act=VPHswish,  s=1,  wm=wm, beta=betas[9], alpha=alpha),#10
            Bneck(nin=96,  nout=96,  k_size=5, exp_size=576, se=True,  act=VPHswish,  s=1,  wm=wm, beta=betas[10], alpha=alpha) #11
        )
        self.conv_2 = nn.Sequential(
            WSConv2D(int(96*wm), int(576*wm), 1, bias=False),
            VPHswish()
        )
        self.conv_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            WSConv2D(int(576*wm), 1024, 1),
            VPHswish(),
            nn.Dropout(p=drop_prob)
        )
        self.conv_4 = WSConv2D(1024, 100, 1)

    def forward(self, input):
        x = self.conv_0(input)
        x = self.bottlenecks_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x.view(x.shape[0], -1)
    
    def name(self):
        return "NFMobilenet_v3_small_" + str(self.wm)