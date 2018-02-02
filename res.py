import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Step(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stri = 1):
        super(Step, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size = 3, padding = 1, stride = stri, bias = False)
        self.bn3 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size = 1, bias = False)
        self.conv_sp = None
        if stri == 2 or in_ch != out_ch:
            self.conv_sp = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1, stride = stri, bias = False),
                #nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)        
        if self.conv_sp != None:
            x = self.conv_sp(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = F.relu(out)
        out = self.conv3(out)

        out.add_(x)
        return out

def getLayer(in_ch, mid_ch, out_ch, n, first_stri = 2):
    layer = []
    layer.append(Step(in_ch, mid_ch, out_ch, first_stri))
    
    for i in range(n - 1):
        layer.append(Step(out_ch, mid_ch, out_ch))

    return nn.Sequential(*layer)

class ResNet_cifar(nn.Module):
    def __init__(self, n):
        super(ResNet_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, padding = 1, bias = False)
        # self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = getLayer(16, 16, 64, n, 1)
        self.layer2 = getLayer(64, 32, 128, n)
        self.layer3 = getLayer(128, 64, 256, n)
        self.bn = nn.BatchNorm2d(256)
        self.linear = nn.Linear(256, 10)
        
        # msra init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size = 8)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
