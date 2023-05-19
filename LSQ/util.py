import torch
from torch import Tensor, nn

# ref : https://github.com/zhutmost/lsq-net

# `(y - y_grad).detach() + y_grad` = redefine backward function, but might be faster
# use former in this implementation but the latter in other models
def grad_scale(x,s):
    y = x
    y_grad = x * s
    return (y - y_grad).detach() + y_grad

# STN
def round_pass(x):
    y = torch.round(x)
    y_grad = x
    return (y - y_grad).detach() + y_grad

class quantizer(nn.Module):
    # v: input
    # s: step size (learnable)
    # p: quantization bits
    # isAct: True if v is activation tensor (so Q_N = 0)
    def __init__(self, p, isAct) -> None:
        super().__init__()
        self.p = p
        self.isAct = isAct
        if isAct:
            self.Qn = 0
            self.Qp = 2**p - 1
        else:
            self.Qn = -2**(p - 1)
            self.Qp = 2**(p - 1) - 1
        self.s = nn.Parameter(torch.ones(1))
        self.s_init = False
        # not init, but should be defined for optimizer

    def forward(self, v: Tensor) -> Tensor:
        if not self.s_init:
            # initialization
            self.s = nn.Parameter(v.detach().abs().mean() * 2 / (self.Qp ** 0.5))
            self.s_init = True
        grad_scale_factor = 1 / ((self.Qp * v.numel()) ** 0.5)
        s = grad_scale(self.s,grad_scale_factor)
        v = v / s
        v = torch.clip(v, self.Qn, self.Qp)
        vbar = round_pass(v)
        vhat = vbar * s
        return vhat
        

def get_qconv2d(bitW):

    class qconv2d(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
            super(qconv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            self.q = quantizer(bitW, isAct=False)

        def forward(self, x: Tensor) -> Tensor:
            quantized_weight = self.q(self.weight)
            return nn.functional.conv2d(x, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
    return qconv2d

def get_qlinear(bitW):

    class qlinear(nn.Linear):
        def __init__(self, in_features, out_features, bias=True):
            super(qlinear, self).__init__(in_features, out_features, bias)
            self.q = quantizer(bitW, isAct=False)

        def forward(self, x):
          quantized_weight = self.q(self.weight)
          return nn.functional.linear(x, quantized_weight, self.bias)

    return qlinear

def get_qReLU(bitA):

    class qReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = quantizer(bitA, isAct=True)

        def forward(self, x):
            return self.q(x)

    return qReLU