import torch.nn as nn
from torch import Tensor

LAYERS = [3, 3, 3]
GROUPS = 2

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=2):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.GroupNorm(groups, out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ClientModel(nn.Module):    
    def __init__(self, lr:float, num_classes:int, device:int):
        super(ClientModel, self).__init__()
        self.groups = GROUPS
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.GroupNorm(self.groups, 16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, LAYERS[0])
        self.layer2 = self.make_layer(32, LAYERS[1], 2)
        self.layer3 = self.make_layer(64, LAYERS[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.GroupNorm(self.groups, out_channels))
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample, self.groups))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, None, self.groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
