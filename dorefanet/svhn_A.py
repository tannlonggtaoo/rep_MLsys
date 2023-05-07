import torch
import torch.nn as nn
from util import *

class SVHN_A(nn.Module): # Model A in paper
    def __init__(self, n_class, bitW, bitA, bitG) -> None:
        super().__init__()
        qconv2d = get_qconv2d(bitW)
        # qlinear = get_qlinear(bitW) # not used in this model
        qReLU = get_qReLU(bitA)
        qgrad = get_qgrad(bitG)

        # input.shape = [batchsize, 3, 40, 40] (REESAMPLED)
        self.l0 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=48,kernel_size=5),
            # 36*36
            nn.MaxPool2d(kernel_size=2,stride=2),
            # 18*18
            # no quantization @ layer 0 except activision
            qReLU(True)
            # 18*18
        )
        
        self.l1 = nn.Sequential(
            qconv2d(in_channels=48,out_channels=64,kernel_size=3,padding=1),
            qgrad(),
            nn.BatchNorm2d(num_features=64,eps=1e-4,momentum=0.9)
            # 18*18
        )
        
        self.l2 = nn.Sequential(
            qconv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
            qgrad(),
            nn.BatchNorm2d(num_features=64,eps=1e-4,momentum=0.9),
            nn.MaxPool2d(kernel_size=2,stride=2),
            qReLU(True)
            # 9*9
        )
        
        self.l3 = nn.Sequential(
            qconv2d(in_channels=64,out_channels=128,kernel_size=3),
            qgrad(),
            nn.BatchNorm2d(num_features=128,eps=1e-4,momentum=0.9),
            qReLU(True)
            # 7*7
        )
        
        self.l4 = nn.Sequential(
            qconv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            qgrad(),
            nn.BatchNorm2d(num_features=128,eps=1e-4,momentum=0.9),
            qReLU(True)
            # 7*7
        )
        
        self.l5 = nn.Sequential(
            qconv2d(in_channels=128,out_channels=128,kernel_size=3),
            qgrad(),
            nn.BatchNorm2d(num_features=128,eps=1e-4,momentum=0.9),
            qReLU()
            # 5*5
        )
        
        self.l6 = nn.Sequential(
            nn.Dropout2d(),
            qconv2d(in_channels=128,out_channels=512,kernel_size=5),
            qgrad(),
            nn.BatchNorm2d(num_features=512,eps=1e-4,momentum=0.9)
        )
        
        self.fc = nn.Sequential(
            # no quantization @ last layer
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512,out_features=n_class)
        )
        
    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = x.squeeze()
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    model = SVHN_A(10,4,5,6)
    x = torch.rand(size=[25,3,40,40])
    y = model(x)