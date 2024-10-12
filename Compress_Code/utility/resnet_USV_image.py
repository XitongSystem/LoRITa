import torch
from torchvision.models import resnet18, ResNet18_Weights

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class conv2d_usv(nn.Module):
    def __init__(self, size_in, size_out, kernel_size=3, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.orig_shape = (size_out, size_in, kernel_size, kernel_size)
        self.kernel_size = None # this is just a holder to be compatible with original implementation
        
        # this initialization is just a holder
        U = torch.Tensor(size_in, size_out)
        self.U = nn.Parameter(U, requires_grad=True)

        V = torch.Tensor(size_in, size_out)
        self.V = nn.Parameter(V, requires_grad=True)

        S = torch.Tensor(size_in)
        self.S = nn.Parameter(S, requires_grad=True)

        self.bias = None 
        if bias:
            bias = torch.Tensor(size_out)
            self.bias = nn.Parameter(bias)

    def forward(self, x):
        weight = (self.U @ torch.diag(self.S) @ self.V).view(self.orig_shape)
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding) 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, factor=1):
        super(BasicBlock, self).__init__()

        if factor == 1:
            self.conv1 = conv2d_usv(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = OrderedDict()
            self.conv1['compress_0'] = conv2d_usv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            for i in range(1, factor):
                self.conv1['compress_'+str(i)] = conv2d_usv(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1 = nn.Sequential(self.conv1)
        
        if factor == 1:
            self.conv2 = conv2d_usv(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        else:
            self.conv2 = OrderedDict()
            self.conv2['compress_0'] = conv2d_usv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            for i in range(1, factor):
                self.conv2['compress_'+str(i)] = conv2d_usv(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv2 = nn.Sequential(self.conv2)

        
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if factor == 1:
                self.shortcut = nn.Sequential(
                    conv2d_usv(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                shortcut = []
                for i in range(factor-1):
                    shortcut.append(conv2d_usv(in_planes, in_planes, kernel_size=1, stride=1, bias=False))
                shortcut.append(conv2d_usv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))
                shortcut.append(nn.BatchNorm2d(self.expansion*planes))
                self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, factor=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if factor == 1:
            self.conv1 = conv2d_usv(3, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
        else:
            self.conv1 = OrderedDict()
            self.conv1['compress_0'] = conv2d_usv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            for i in range(1,factor):
                self.conv1['compress_'+str(i)] = conv2d_usv(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1 = nn.Sequential(self.conv1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, factor=factor)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, factor=factor)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, factor=factor)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, factor=factor)
           
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, factor):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, factor))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_USV_image(num_classes=10, factor=1):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, factor=factor)
