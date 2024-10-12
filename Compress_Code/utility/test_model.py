import torch
import numpy as np
import torch.nn as nn 
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def get_test_acc(model, args, weight_decay=0.0, scheduler=False, ratio=1, testing=True):
    device = next(model.parameters()).device
    if args.dataset.lower() == 'mnist':
        testset = MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor()]))
        trainset = MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ]))
        img_size = 28
        n_channels = 1 
    elif args.dataset.lower() == 'cifar10':
        if args.no_aug and ratio < 1.0:
            testset = CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703225141799082, 0.24348516474564, 0.26158783926049628))
                ]))
            trainset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703225141799082, 0.24348516474564, 0.26158783926049628))
                ]))
        else:
            testset = CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703225141799082, 0.24348516474564, 0.26158783926049628))
                ]))
            trainset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703225141799082, 0.24348516474564, 0.26158783926049628))
                ]))

        img_size = 32
        n_channels = 3
    else:
        if args.no_aug and ratio < 1.0:
            testset = CIFAR100(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703225141799082, 0.24348516474564, 0.26158783926049628))
            ]))
            trainset = CIFAR100(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703225141799082, 0.24348516474564, 0.26158783926049628))
            ]))
        else:
            testset = CIFAR100(root='./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]))
            trainset = CIFAR100(root='./data', train=True, download=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]))

        img_size = 32
        n_channels = 3

    if ratio < 1.0:
        trainset = torch.utils.data.Subset(trainset, np.arange(int(ratio*len(trainset))) )
        print("subset: ", len(trainset))

    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    if testing:
        with torch.no_grad():
            losses, labels, preds = [], [], []
            model.eval()
            for x, y in tqdm(testloader):
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                pred = y_hat.argmax(dim=1)
                l = loss(y_hat, y)
                losses.append(l.item())
                preds.extend(pred.cpu().numpy())
                labels.extend(y.cpu().numpy())

            acc = (np.stack(preds, axis=0) == np.stack(labels, axis=0)).sum()/ len(testset)
            print(f'test_Loss: {np.array(losses).mean():.4f}| test_Accuracy: {acc:.4f}')
    else:
        test_train = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
        test_train_cross = nn.CrossEntropyLoss()
        with torch.no_grad():
            losses, labels, preds = [], [], []
            model.eval()
            for x, y in tqdm(test_train):
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                pred = y_hat.argmax(dim=1)
                l = test_train_cross(y_hat, y)
                losses.append(l.item())
                preds.extend(pred.cpu().numpy())
                labels.extend(y.cpu().numpy())

            acc = (np.stack(preds, axis=0) == np.stack(labels, axis=0)).sum()/ len(trainset)
            print(f'train_Loss: {np.array(losses).mean():.4f}| train_Accuracy: {acc:.4f}')

    if args.finetune and args.reference_acc > (acc + 0.01):
        #try:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=weight_decay)

        if scheduler:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, threshold=1e-2)

        tor = 0
        best_acc = 0.0
        for epoch in range(50):
            model.train()

            correct, samples = 0, 0
            for x, y in tqdm(trainloader):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                y_hat = model(x)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()

                correct += (torch.argmax(y_hat.data, 1) == y).sum().item()
                samples += x.size(0)

            acc = correct/samples
            if scheduler:
                scheduler.step(acc)

            if best_acc < acc - 0.005:
                best_acc = acc 
                tor = 0
            else:
                tor += 1

            if tor > 5:
                break
            print("finetune acc:", epoch, best_acc, acc)

        # pred
        with torch.no_grad():
            losses, labels, preds = [], [], []
            model.eval()
            for x, y in tqdm(testloader):
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                pred = y_hat.argmax(dim=1)
                l = loss(y_hat, y)
                losses.append(l.item())
                preds.extend(pred.cpu().numpy())
                labels.extend(y.cpu().numpy())

            acc = (np.stack(preds, axis=0) == np.stack(labels, axis=0)).sum()/ len(testset)
            print(f'test_Loss: {np.array(losses).mean():.4f}| test_Accuracy: {acc:.4f}')

        # except:
        #     pass
    if testing:
        return acc
    else:
        return acc, np.array(losses).mean()

