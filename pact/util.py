# ref: https://github.com/KwangHoonAn/PACT
import torch
from torch import nn

class pact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, bitA):
        ctx.save_for_backward(x, alpha)
        y = torch.clip(x, min=0, max=alpha.item())
        s = (2 ** bitA - 1) / alpha
        y_q = torch.round(y * s) / s
        return y_q

    @staticmethod
    def backward(ctx, grad_outputs):
        x, alpha = ctx.saved_tensors
        x_mask = (~((x < 0) | (x > alpha))).float()
        grad_alpha = torch.sum(grad_outputs * torch.ge(x, alpha).float()).view(-1)
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_outputs*x_mask, grad_alpha, None
    
# following are the same as dorefanet so just copied (except grad quantization)

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