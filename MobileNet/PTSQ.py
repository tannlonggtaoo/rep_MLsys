import torch
from torch import nn
from torch.ao import quantization
from naivemodel import NaiveMobileNetV2
from train import getloader,test,loadparam
from torch.utils.data import DataLoader,Subset
import copy
device = torch.device("cpu")

def PTSQ(model, activision_nbit, weight_nbit, calib_loader) -> nn.Module:
    # STEP 1 fusion
    newmodel = copy.deepcopy(model)
    newmodel.set_nbit(activision_nbit,weight_nbit)
    newmodel.fuse_model()

    # STEP 2 observing
    model_prepared = quantization.prepare(newmodel)
    model_prepared.to(device)
    model_prepared.eval()

    print('Calibrating...')
    test(model_prepared, device, calib_loader, verbose=False)
    print(f'Calibration done...')

    # STEP 3 convert
    quantization.convert(model_prepared, inplace=True)
    return model_prepared

if __name__ == '__main__':
    model = NaiveMobileNetV2(n_class=10,quantizable=True)
    ckpt_path = './MobileNet/models/original'
    ckpt, _ = loadparam(ckpt_path)
    model.load_state_dict(ckpt)

    train_loader, test_loader = getloader(96,0)
    calib_loader = DataLoader(Subset(train_loader.dataset,list(range(10000))),96)

    for activision_nbit in [2,3,4,5,6]:
        for weight_nbit in [2,4,6]:
            print(f'[main] testing ptsq : activision {activision_nbit} bit, weight {weight_nbit} bit')
            qmodel = PTSQ(model, activision_nbit, weight_nbit, calib_loader)
            test(qmodel,device,test_loader)