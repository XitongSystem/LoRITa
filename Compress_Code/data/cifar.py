import torch, os
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Cifar100:
    def __init__(self, batch_size, threads, size=(32, 32), augmentation=False):
        mean, std = self._get_statistics()

        if augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                #transforms.RandomCrop(32, padding=4),
                #torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)
        self.val = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class Cifar10:
    def __init__(self, batch_size, threads, size=(32, 32), augmentation=False):

        mean, std = self._get_statistics()
        if augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        
        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)
        self.val = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


class TinyImageNet:
    def __init__(self, batch_size, threads, size=(64, 64), augmentation=False):
        mean = np.array([ 0.485, 0.456, 0.406 ])
        std = np.array([ 0.229, 0.224, 0.225 ])

        data_dir = '../pytorch-tiny-imagenet/tiny-imagenet-200/'
        if augmentation:
            train_transform = transforms.Compose([
                torchvision.transforms.Resize(size=size),
                transforms.RandomCrop(size[0], padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                torchvision.transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            torchvision.transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        val_transform = transforms.Compose([
            torchvision.transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        data_transforms = {}
        data_transforms['train'] = train_transform
        data_transforms['val'] = val_transform
        data_transforms['test'] = test_transform

        image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val','test']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                     shuffle=False, num_workers=threads)
                      for x in ['test', 'val']}
        dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                                     shuffle=True, num_workers=threads)

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

        self.train = dataloaders['train']
        self.test = dataloaders['test']
        self.val = dataloaders['val']