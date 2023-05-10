import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from resnet20 import NaiveResnet20
import os
import re
import numpy as np

def getloader(batch_size, num_workers, mode='both'):
    # mode = 'train' 'test' or 'both'
    if (mode == 'both') or (mode == 'train'):
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # computed from cifar10 trainset
        ])

        trainset = datasets.CIFAR10(root='.\dataset\CIFAR10', train=True, download=True, transform=transform_train)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if (mode == 'both') or (mode == 'test'):
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = datasets.CIFAR10(root='.\dataset\CIFAR10', train=False, download=True, transform=transform_test)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # normalization is already done...
    if mode == 'both':
        return trainloader, testloader
    if mode == 'train':
        return trainloader
    if mode == 'test':
        return testloader
    
def train(model, device, train_loader, optimizer, epoch, loss_fn, lamda):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        reg = 0.0
        for name,param in model.named_parameters():
            if "alpha" in name:
                reg += torch.pow(param, 2)
        loss += lamda * reg
        loss.backward()
        optimizer.step()
        if(batch_idx + 1)%(len(train_loader) // 3) == 0: 
            print(f'[Train] Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

def test(model, device, test_loader, loss_fn = nn.CrossEntropyLoss(reduction='sum'), verbose = True):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if verbose:
        print(f'[Test] Average loss: {test_loss:.3f} Accuracy: {correct}/{len(test_loader.dataset)} ({100 * correct / len(test_loader.dataset):.3f}%)\n')
    return test_loss

def loadparam(ckpt_path):
    files = os.listdir(ckpt_path)
    epochs = [int(re.findall(r'epoch[0-9]*_', file)[0][5:-1]) for file in files]
    load_path = os.path.join(ckpt_path, files[np.argmax(epochs)])
    begin_epoch = max(epochs) + 1
    ckpt = torch.load(load_path)
    return ckpt, begin_epoch

def pact_examine(bitW,bitA):

    batch_size = 128 # a little different
    n_epoch = 140
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = getloader(batch_size,num_workers)

    model = NaiveResnet20(n_class=10,bitW=bitW,bitA=bitA)
    model.to(device)
    resume_if_ckpt_exists = False
    begin_epoch = 0 # will be overwritten if previous ckpt is loaded
    ckpt_path = './pact/models/resnet20'

    if (resume_if_ckpt_exists):
        ckpt, begin_epoch = loadparam(ckpt_path)
        model.load_state_dict(ckpt)

    folder_path = os.path.join(ckpt_path, f'W{bitW}_A{bitA}')
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120])
    loss_fn = nn.CrossEntropyLoss()
    loss_fn_test = nn.CrossEntropyLoss(reduction='sum')
    lamda = 0.0002

    for epoch in range(begin_epoch, begin_epoch + n_epoch):
        train(model, device, train_loader, optimizer, epoch, loss_fn, lamda)
        testloss = test(model, device, test_loader, loss_fn_test)
        
        torch.save(model.state_dict(), os.path.join(folder_path, f'epoch{epoch}_loss{testloss:.6f}.pth'))
        print(f"[sys] Model saved @ epoch {epoch}...\n")
        scheduler.step()

if __name__ == "__main__":
    #for bitW,bitA in [[6,6],[5,5],[4,4],[3,3],[2,2]]:
    for bitW,bitA in [[6,6]]:
        print(f'[sys] bitW {bitW}, bitA {bitA} ------------------------------------')
        pact_examine(bitW,bitA)