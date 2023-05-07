import torch
import torch.nn as nn
from torch.nn.quantized import FloatFunctional
from torch.ao import quantization

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, quantizable = False):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.quantizable = quantizable
        if quantizable:
            self.arith = FloatFunctional()

    def forward(self, x):
        if self.quantizable:
            return self.arith.add_relu(self.residual_function(x), self.shortcut(x))
        else:
            return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
class NaiveResnet(nn.Module):
    # similar to resnet18
    def __init__(self, n_class, quantizable = False) -> None:
        super().__init__()
        self.quantizable = quantizable

        # inputshape = [batchsize, 3, 32, 32]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        # [b, 64, 32, 32]
        self.conv2 = nn.Sequential(BasicBlock(64,64,1,quantizable), BasicBlock(64,64,1,quantizable))
        self.conv3 = nn.Sequential(BasicBlock(64,128,2,quantizable), BasicBlock(128,128,1,quantizable))
        self.conv4 = nn.Sequential(BasicBlock(128,256,2,quantizable), BasicBlock(256,256,1,quantizable))
        self.conv5 = nn.Sequential(BasicBlock(256,512,2,quantizable), BasicBlock(512,512,1,quantizable))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_class)

        if self.quantizable:
            # default quantization setting, use set_nbit() to overwrite
            self.quant = quantization.QuantStub()
            self.dequant = quantization.DeQuantStub()

    def set_nbit(self, activition_nbit, weight_nbit):
        if not self.quantizable:
            print("[Resnet18] Not quantizable...")
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
        self.conv1 = quantization.fuse_modules(self.conv1,[['0','1','2']])
        conv = [self.conv2,self.conv3,self.conv4,self.conv5]
        for i in range(len(conv)):
            for j in range(len(conv[i])):

                conv[i][j] = quantization.fuse_modules(conv[i][j],[['residual_function.0','residual_function.1','residual_function.2'],['residual_function.3','residual_function.4']])

                if len(conv[i][j].shortcut) == 2: conv[i][j] = quantization.fuse_modules(conv[i][j],['shortcut.0','shortcut.1'])

    def forward(self, x):
        if self.quantizable:
            x = self.quant(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.squeeze([2,3])
        x = self.fc(x)
        if self.quantizable:
            x = self.dequant(x)
        return x

if __name__ == "__main__":
    # for debugging
    # data = torch.rand([10,3,32,32])
    # net = NaiveResnet(7)
    # out = net(data)
    # print(net)
    # print(data.shape)
    # print(out.shape)

    # for quantizable debugging
    qnet = NaiveResnet(7,True)
    # load ckpt if exists...
    qnet.set_nbit(2,2)
    qnet.fuse_model()
    print(qnet)