import torch
import torch.nn as nn
from torch.nn.quantized import FloatFunctional
from torch.ao import quantization

class blk_stride(nn.Module):
    def __init__(self, in_channels, out_channels, t, stride, quantizable = False) -> None:
        super().__init__()
        self.stride = stride
        expanded_channels = in_channels * t
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True)
        )
        self.dwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False),
            # note groups work on channel dim
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

        self.quantizable = quantizable
        if quantizable and (stride == 1):
            self.arith = FloatFunctional()
            # must be used during calib otherwise the observer on this object get nothing to update min & max
        
    def forward(self, x):
        r = self.conv1(x)
        r = self.dwise(r)
        r = self.conv2(r)
        if self.stride != 1:
            return r
        # else
        if self.quantizable:
            return self.arith.add(r, self.shortcut(x))
        else:
            return r + self.shortcut(x)
    
class NaiveMobileNetV2(nn.Module):
    def __init__(self, n_class, quantizable = False) -> None:
        super().__init__()
        self.quantizable = quantizable

        # [b,3,32,32]
        self.inconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        ) # stride -> 1
        # [b,32,32,32]
        self.l1 = blk_stride(32,16,1,1,quantizable)
        # [b,16,32,32]
        self.l2 = nn.Sequential(
            blk_stride(16,24,6,1,quantizable),
            blk_stride(24,24,6,1,quantizable),
        ) # stride -> 1
        # [b,24,32,32]
        self.l3 = nn.Sequential(
            blk_stride(24,32,6,2,quantizable),
            blk_stride(32,32,6,1,quantizable),
            blk_stride(32,32,6,1,quantizable),
        )
        # [b,32,16,16]
        self.l4 = nn.Sequential(
            blk_stride(32,64,6,2,quantizable),
            blk_stride(64,64,6,1,quantizable),
            blk_stride(64,64,6,1,quantizable),
            blk_stride(64,64,6,1,quantizable),
        )
        # [b,64,8,8]
        self.l5 = nn.Sequential(
            blk_stride(64,96,6,1,quantizable),
            blk_stride(96,96,6,1,quantizable),
            blk_stride(96,96,6,1,quantizable),
        )
        # [b,96,8,8]
        self.l6 = nn.Sequential(
            blk_stride(96,160,6,2,quantizable),
            blk_stride(160,160,6,1,quantizable),
            blk_stride(160,160,6,1,quantizable),
        )
        # [b,160,4,4]
        self.l7 = blk_stride(160,320,6,1,quantizable)
        # [b,320,4,4]
        self.outconv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        # [b,1280,4,4]
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        # [b,1280,1,1]
        self.fc = nn.Linear(1280, n_class)

        if self.quantizable:
            self.quant = quantization.QuantStub()
            self.dequant = quantization.DeQuantStub()

    def set_nbit(self, activition_nbit, weight_nbit):
        if not self.quantizable:
            print("[MobileNetV2] Not quantizable...")
            return
        self.qconfig = quantization.QConfig(
            activation=quantization.observer.MinMaxObserver.with_args(quant_min=0,
                                                                      quant_max= 2**activition_nbit - 1),
            weight=quantization.observer.MinMaxObserver.with_args(quant_min = -(2**(weight_nbit-1)-1),
                                                                  quant_max = 2**(weight_nbit-1)-1,
                                                                  dtype=torch.qint8,
                                                                  qscheme=torch.per_tensor_symmetric))
        self.quant = quantization.QuantStub(self.qconfig)
        self.dequant = quantization.DeQuantStub(self.qconfig)

    def fuse_model(self):
        self.eval()
        # not fuser method for ReLU6
        self.inconv = quantization.fuse_modules(self.inconv,[['0','1']])
        self.outconv = quantization.fuse_modules(self.outconv,[['0','1']])

        blk = [self.l1,self.l7]
        for i in range(len(blk)):
            blk[i] = quantization.fuse_modules(blk[i],[['conv1.0','conv1.1'],['dwise.0','dwise.1'],['conv2.0','conv2.1']])
            if len(blk[i].shortcut) == 2: blk[i] = quantization.fuse_modules(blk[i],['shortcut.0','shortcut.1'])
        conv = [self.l2,self.l3,self.l4,self.l5,self.l6]
        for i in range(len(conv)):
            for j in range(len(conv[i])):

                conv[i][j] = quantization.fuse_modules(conv[i][j],[['conv1.0','conv1.1'],['dwise.0','dwise.1'],['conv2.0','conv2.1']])
                if len(conv[i][j].shortcut) == 2: conv[i][j] = quantization.fuse_modules(conv[i][j],['shortcut.0','shortcut.1'])

    def forward(self, x):
        if self.quantizable:
            x = self.quant(x)
        x = self.inconv(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.outconv(x)
        x = self.avgpool(x)
        x = x.squeeze([2,3])
        x = self.fc(x)
        if self.quantizable:
            x = self.dequant(x)
        return x

if __name__ == "__main__":
    # for debugging
    data = torch.rand([10,3,32,32])
    # net = NaiveMobileNetV2(7)
    # out = net(data)
    # print(net)
    # print(data.shape)
    # print(out.shape)

    # for quantizable debugging
    qnet = NaiveMobileNetV2(7,True)
    # load ckpt if exists...
    qnet.set_nbit(2,2)
    qnet.fuse_model()
    print(qnet)
