import torch, math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

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

class VGG_USV(nn.Module):
    def __init__(self, vgg_name, num_classes=10, factor=1):
        super(VGG_USV, self).__init__()
        self.factor=factor
        self.features = self._make_layers(cfg[vgg_name])
        
        if factor == 1:
            self.classifier = linear_usv(512, num_classes)
        else:
            self.classifier = OrderedDict()
            for i in range(factor - 1):
                self.classifier['compress_'+str(i)] = linear_usv(512, 512, bias=False)
            self.classifier['compress_'+str(factor-1)] = linear_usv(512, num_classes)
            self.classifier = nn.Sequential(self.classifier)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = OrderedDict() 
        in_channels = 3
        
        for i, x in enumerate(cfg):
            if x == 'M':
                layers['max_'+str(i)] = nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                if self.factor == 1:
                    layers['conv_'+str(i)] = conv2d_usv(in_channels, x, kernel_size=3, padding=1, bias=False)
                else:
                    layers['conv_'+str(i)+':compress_0'] = conv2d_usv(in_channels, x, kernel_size=3, padding=1, bias=False)
                    for k in range(1, self.factor):
                        layers['conv_'+str(i)+':compress_'+str(k)] = conv2d_usv(x, x, kernel_size=1, padding=0, bias=False) 
                    
                layers['bn_'+str(i)] = nn.BatchNorm2d(x)
                layers['relu_'+str(i)] = nn.ReLU()
                in_channels = x

        layers['avg'] = nn.AvgPool2d(kernel_size=1, stride=1)
        
        return nn.Sequential(layers)