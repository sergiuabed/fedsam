from typing import Any, Callable, List, Optional, Type, Union
from .resnet import BasicBlock, Bottleneck, conv1x1, conv3x3

import torch
import torch.nn as nn
from torch import Tensor

LAYERS = [2, 2, 2, 3]
BLOCK = BasicBlock
NORM_LAYER = nn.GroupNorm
ZERO_INIT_RESIDUAL = False
GROUPS = 1
WIDTH_PER_GROUP = 64
IN_PLANES = 64
BASE_WIDTH = 64


class ClientModel(nn.Module):    
    def __init__(self, lr:float, num_classes:int, device:int):
        super(ClientModel, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.lr = lr
        self.in_planes = IN_PLANES
        
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = NORM_LAYER(GROUPS, self.in_planes) # NOT 100% 'GROUPS' SHOULD BE PASSED HERE
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BLOCK, 64, LAYERS[0])
        self.layer2 = self._make_layer(BLOCK, 128, LAYERS[1], stride=2)
        self.layer3 = self._make_layer(BLOCK, 256, LAYERS[2], stride=2)
        self.layer4 = self._make_layer(BLOCK, 512, LAYERS[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BLOCK.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if ZERO_INIT_RESIDUAL:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                NORM_LAYER(GROUPS, planes * block.expansion), # ADDED 'GROUPS'
            )

        layers = []
        layers.append(
            block(
                self.in_planes,
                planes,
                stride,
                downsample,
                GROUPS,
                BASE_WIDTH,
                NORM_LAYER,
            )
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.in_planes,    # added "inplanes="
                    planes=planes,              # added "planes="
                    groups=GROUPS,
                    base_width=BASE_WIDTH,
                    norm_layer=NORM_LAYER,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size


