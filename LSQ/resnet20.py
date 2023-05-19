import torch
import torch.nn as nn
from util import *

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, bitW, bitA):
        super().__init__()
        qconv = get_qconv2d(bitW)
        qReLU = get_qReLU(bitA)

        self.res0 = nn.Sequential(
            qconv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            qReLU()
        )
        self.res1 = nn.Sequential(
            qconv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                qconv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.res1(self.res0(x))
        return out + self.shortcut(x)
    
class NaiveResnet20(nn.Module):
    # similar to resnet20
    def __init__(self, n_class, bitW, bitA) -> None:
        super().__init__()

        # inputshape = [batchsize, 3, 32, 32]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        # [b, 64, 32, 32]
        self.conv2 = nn.Sequential(BasicBlock(64,64,1,bitW,bitA), BasicBlock(64,64,1,bitW,bitA))
        self.conv3 = nn.Sequential(BasicBlock(64,128,2,bitW,bitA), BasicBlock(128,128,1,bitW,bitA))
        self.conv4 = nn.Sequential(BasicBlock(128,256,2,bitW,bitA), BasicBlock(256,256,1,bitW,bitA))
        self.conv5 = nn.Sequential(BasicBlock(256,512,2,bitW,bitA), BasicBlock(512,512,1,bitW,bitA))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.squeeze([2,3])
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # for debugging
    data = torch.rand([10,3,32,32])
    net = NaiveResnet20(11,bitW=4,bitA=6)
    out = net(data)
    print(net)
    print(data.shape)
    print(out.shape)