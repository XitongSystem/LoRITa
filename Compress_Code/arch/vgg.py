import torch, math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG16_Compress': [4, 14, 'M', 18, 29, 'M', 39, 35, 36, 'M', 33, 21, 14, 'M', 9, 9, 21, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, input_res=32, factor=1):
        super(VGG, self).__init__()
        self.factor=factor
        self.features = self._make_layers(cfg[vgg_name])
        
        hidden=512
        if input_res != 32:
            hidden=512*4

        if 'Compress' in vgg_name:
            hidden = cfg[vgg_name][-2]
            
        if factor == 1:
            self.classifier = nn.Linear(hidden, num_classes)
        else:
            self.classifier = OrderedDict()
            for i in range(factor - 1):
                self.classifier['compress_'+str(i)] = nn.Linear(hidden, hidden, bias=False)
            self.classifier['compress_'+str(factor-1)] = nn.Linear(hidden, num_classes)
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
                    layers['conv_'+str(i)] = nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False)
                else:
                    layers['conv_'+str(i)+':compress_0'] = nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False)
                    for k in range(1, self.factor):
                        layers['conv_'+str(i)+':compress_'+str(k)] = nn.Conv2d(x, x, kernel_size=1, padding=0, bias=False) 
                    
                layers['bn_'+str(i)] = nn.BatchNorm2d(x)
                layers['relu_'+str(i)] = nn.ReLU()
                in_channels = x

        layers['avg'] = nn.AvgPool2d(kernel_size=1, stride=1)
        
        return nn.Sequential(layers)