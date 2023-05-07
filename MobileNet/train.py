import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from naivemodel import NaiveMobileNetV2
import os
import re
import numpy as np

def getloader(batch_size, num_workers):
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # computed from cifar10 trainset
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='.\dataset\CIFAR10', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = datasets.CIFAR10(root='.\dataset\CIFAR10', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # normalization is already done...
    return trainloader, testloader
    
def train(model, device, train_loader, optimizer, epoch, loss_fn):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
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

if __name__ == "__main__":
    batch_size = 96
    n_epoch = 50
    num_workers = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = getloader(batch_size,num_workers)
    n_class = 10

    model = NaiveMobileNetV2(n_class=10)
    model.to(device)
    resume_if_ckpt_exists = True
    begin_epoch = 0 # will be overwritten if previous ckpt is loaded
    ckpt_path = './MobileNet/models/original'

    if (resume_if_ckpt_exists):
        ckpt, begin_epoch = loadparam(ckpt_path)
        model.load_state_dict(ckpt)


    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn_test = nn.CrossEntropyLoss(reduction='sum')

    for epoch in range(begin_epoch, begin_epoch + n_epoch):
        train(model, device, train_loader, optimizer, epoch, loss_fn)
        testloss = test(model, device, test_loader, loss_fn_test)
        torch.save(model.state_dict(), os.path.join(ckpt_path, f'mobilenetv2_epoch{epoch}_loss{testloss:.6f}.pth'))
        print(f"[sys] Model saved @ epoch {epoch}...\n")
        scheduler.step(testloss)