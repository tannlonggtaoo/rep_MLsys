import torch
import torch.nn as nn
from util import *

class blk_stride(nn.Module):
    def __init__(self, in_channels, out_channels, t, stride, bitW, bitA) -> None:
        super().__init__()
        self.stride = stride
        expanded_channels = in_channels * t
        qconv = get_qconv2d(bitW)
        qReLU = get_qReLU(bitA)

        self.conv1 = nn.Sequential(
            qconv(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expanded_channels),
            qReLU()
        )
        self.dwise = nn.Sequential(
            qconv(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            qReLU()
        )
        self.conv2 = nn.Sequential(
            qconv(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                qconv(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.dwise(x1)
        x3 = self.conv2(x2)
        if self.stride != 1:
            return x3
        else:
            return x3 + self.shortcut(x)
    
class NaiveMobileNetV2(nn.Module):
    def __init__(self, n_class, bitW, bitA) -> None:
        super().__init__()

        # [b,3,32,32]
        self.inconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        ) # stride -> 1
        # [b,32,32,32]
        self.l1 = blk_stride(32,16,1,1, bitW, bitA)
        # [b,16,32,32]
        self.l2 = nn.Sequential(
            blk_stride(16,24,6,1, bitW, bitA),
            blk_stride(24,24,6,1, bitW, bitA),
        ) # stride -> 1
        # [b,24,32,32]
        self.l3 = nn.Sequential(
            blk_stride(24,32,6,2, bitW, bitA),
            blk_stride(32,32,6,1, bitW, bitA),
            blk_stride(32,32,6,1, bitW, bitA),
        )
        # [b,32,16,16]
        self.l4 = nn.Sequential(
            blk_stride(32,64,6,2, bitW, bitA),
            blk_stride(64,64,6,1, bitW, bitA),
            blk_stride(64,64,6,1, bitW, bitA),
            blk_stride(64,64,6,1, bitW, bitA),
        )
        # [b,64,8,8]
        self.l5 = nn.Sequential(
            blk_stride(64,96,6,1, bitW, bitA),
            blk_stride(96,96,6,1, bitW, bitA),
            blk_stride(96,96,6,1, bitW, bitA),
        )
        # [b,96,8,8]
        self.l6 = nn.Sequential(
            blk_stride(96,160,6,2, bitW, bitA),
            blk_stride(160,160,6,1, bitW, bitA),
            blk_stride(160,160,6,1, bitW, bitA),
        )
        # [b,160,4,4]
        self.l7 = blk_stride(160,320,6,1, bitW, bitA)
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

    def forward(self, x):
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
    qnet = NaiveMobileNetV2(7,bitW=4,bitA=5)
    # load ckpt if exists...
    out = qnet(data)
    print(qnet)
