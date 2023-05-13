import torch
import torch.nn as nn

USECUDA = True

# ref: https://github.com/zzzxxxttt/pytorch_DoReFaNet/blob/master/utils/quant_dorefa.py
def get_quantize_fn(k):

    class quantize_fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            if k == 1: 
                return torch.sign(x)
            if k == 32: 
                return x
            # 2 <= k <= 8:
            s = float(2 ** k - 1)
            return torch.round(x * s) / s
        @staticmethod
        def backward(ctx, grad_outputs):
            # STN
            grad_at_input = grad_outputs.clone()
            return grad_at_input
    
    return quantize_fn().apply

def get_id_fn(bitG): # get identity module (with special backward func) for grad

    class id_fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x
        @staticmethod
        def backward(ctx, dr):
            # dr.shape = [batchsize,...]
            dr_abs = torch.abs(dr)
            dr_abs_max0 = torch.amax(dr_abs, dim=list(range(1,dr.ndim)), keepdim=True)
            dr = dr / dr_abs_max0
            s = float(2 ** bitG - 1)
            # Noise
            # if USECUDA:
            # note here to('cuda') after the tensor is generated on CPU will cost sooooooo much time!!!!!!
            # N = (torch.rand(size=dr.shape) - 0.5).to('cuda') / s
            N = (torch.rand(size=dr.shape,device='cuda') - 0.5) / s
            # else:
            #     N = (torch.rand(size=dr.shape) - 0.5) / s
            dr = dr * 0.5 + 0.5 + N
            dr = torch.clip(dr, 0, 1)
            quantize_fn = get_quantize_fn(bitG)
            dr = quantize_fn(dr) - 0.5
            return 2 * dr_abs_max0 * dr
        
    return id_fn().apply

class fw(nn.Module): # quantization module for weights
    def __init__(self, bitW) -> None:
        super().__init__()
        self.bitW = bitW
        self.quantize_fn = get_quantize_fn(bitW)

    def forward(self, x):
        if self.bitW == 32: # no quantization
            return x
        if self.bitW == 1: # BWN
            x_avg = torch.mean(torch.abs(x)).detach()
            return self.quantize_fn(x / x_avg) * x_avg # keeps the abs avg unchanged
        # 2 <= k <= 8:
        x = torch.tanh(x)
        x = (x / torch.max(torch.abs(x))) * 0.5 + 0.5
        return 2 * self.quantize_fn(x) - 1

class fa(nn.Module): # quantization function for ReLU
    def __init__(self, bitA) -> None:
        super().__init__()
        self.bitA = bitA
        self.quantize_fn = get_quantize_fn(bitA)

    def forward(self, x):
        if self.bitA == 32:
            return x
        else:
            return self.quantize_fn(torch.clip(x, 0, 1))
        
class fg(nn.Module): # quantization function for gradients (MAY NOT BE USED)
    def __init__(self, bitG) -> None:
        super().__init__()
        self.bitG = bitG
        self.id_fn = get_id_fn(self.bitG)

    def forward(self, x):
        if self.bitG == 32:
            return x
        return self.id_fn(x)

# below: for exterior call

def get_qgrad(bitG):

    class qgrad(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bitG = bitG
            self.fg = fg(bitG)

        def forward(self, x):
            return self.fg(x)
    
    return qgrad

def get_qconv2d(bitW):

    class qconv2d(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
            super(qconv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            # nn.Conv2d is used as a param recorder
            self.bitW = bitW
            self.fw = fw(bitW)

        def forward(self, x):
            quantized_weight = self.fw(self.weight)
            return nn.functional.conv2d(x, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
    return qconv2d

def get_qlinear(bitW):

    class qlinear(nn.Linear):
        def __init__(self, in_features, out_features, bias=True):
            super(qlinear, self).__init__(in_features, out_features, bias)
            self.bitW = bitW
            self.fw = fw(bitW)

        def forward(self, x):
          quantized_weight = self.fw(self.weight)
          return nn.functional.linear(x, quantized_weight, self.bias)

    return qlinear

def get_qReLU(bitA):

    class qReLU(nn.ReLU):
        def __init__(self, inplace: bool = False):
            super().__init__(inplace)
            self.bitA = bitA
            self.fa = fa(bitA)

        def forward(self, x):
            x = nn.functional.relu(x, self.inplace)
            return self.fa(x)

    return qReLU
    
        