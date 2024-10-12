'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
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

class linear_usv(nn.Module):
    def __init__(self, size_in, size_out, bias=True):
        super().__init__()

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
        weight = (self.U @ torch.diag(self.S) @ self.V)
        return F.linear(x, weight, self.bias) 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, factor=1, option='B'):
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

            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A (ResNet20 Type).
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            else:
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

class ResNet_USV(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, input_res=True, factor=1):
        super(ResNet_USV, self).__init__()
        self.in_planes = 64

        if factor == 1:
            self.conv1 = conv2d_usv(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        else:
            self.conv1 = OrderedDict()
            self.conv1['compress_0'] = conv2d_usv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            for i in range(1,factor):
                self.conv1['compress_'+str(i)] = conv2d_usv(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1 = nn.Sequential(self.conv1)

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, factor=factor)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, factor=factor)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, factor=factor)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, factor=factor)

        if input_res:
            if factor == 1:
                self.linear = linear_usv(512*block.expansion, num_classes)
            else:
                self.linear = OrderedDict()
                for i in range(factor - 1):
                    self.linear['compress_'+str(i)] = linear_usv(512*block.expansion, 512*block.expansion, bias=False)
                self.linear['compress_'+str(factor - 1)] = linear_usv(512*block.expansion, num_classes)
                self.linear = nn.Sequential(self.linear)

        else:
            # for tiny mnist
            if factor == 1:
                self.linear = linear_usv(512*block.expansion*4, num_classes)
            else:
                self.linear = OrderedDict()
                for i in range(factor - 1):
                    self.linear['compress_'+str(i)] = linear_usv(512*block.expansion*4, 512*block.expansion*4, bias=False)
                self.linear['compress_'+str(factor - 1)] = linear_usv(512*block.expansion*4, num_classes)
                self.linear = nn.Sequential(self.linear)

    def _make_layer(self, block, planes, num_blocks, stride, factor):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, factor, option='B'))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

#################################################################################
# For ResNet20/32 Typed Networks
#################################################################################

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, factor = 1):
        super().__init__()
        if factor == 1:
            self.net1 = linear_usv(dim, hidden_dim)

        else:
            self.net1 = OrderedDict()
            for i in range(factor - 1):
                self.net1['compress_'+str(i)] = linear_usv(dim, dim, bias=False)
            self.net1['compress_'+str(factor - 1)] = linear_usv(dim, hidden_dim)
            self.net1 = nn.Sequential(self.net1)

    def forward(self, x):
        x = self.net1(x)
        return x

class ResNet_small_USV(nn.Module):
    # modified from DLRT-Net NIPS 2023
    def __init__(self, block, num_blocks, num_classes=10, input_res=True, factor=1):
        super(ResNet_small_USV, self).__init__()
        self.in_planes = 16
        
        if factor == 1:
            self.conv1 = conv2d_usv(3, 16, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        else:
            self.conv1 = OrderedDict()
            self.conv1['compress_0'] = conv2d_usv(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            for i in range(1,factor):
                self.conv1['compress_'+str(i)] = conv2d_usv(16, 16, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1 = nn.Sequential(self.conv1)

        self.bn1 = nn.BatchNorm2d(16)
        self.layer1,self.layer1_list  = self._make_layer(block, 16, num_blocks[0], stride=1, factor=factor)
        self.layer2,self.layer2_list  = self._make_layer(block, 32, num_blocks[1], stride=2, factor=factor)
        self.layer3,self.layer3_list  = self._make_layer(block, 64, num_blocks[2], stride=2, factor=factor)

        if input_res:
            self.linear = FeedForward(64, num_classes, factor)

        else:
            # for tiny mnist
            self.linear = FeedForward(64*4, num_classes, factor)

    def _make_layer(self, block, planes, num_blocks, stride, factor):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, factor, option='A'))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers),layers

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_USV(num_classes=10, input_res=32, factor=1):
    return ResNet_USV(BasicBlock, [2, 2, 2, 2], num_classes, input_res==32, factor=factor)

def ResNet20_USV(num_classes=10, input_res=32, factor=1):
    return ResNet_small_USV(BasicBlock, [3, 3, 3], num_classes, input_res==32, factor=factor)

def test():
    net = ResNet18Noise(10)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
