"""This file defines the ResNet architecture for CIFAR-10/100 dataset.

Modified from akamaster/pytorch_resnet_cifar10
"""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Building block for ResNet.

    Attributes:
        conv1: First convolutional layer.
        bn1: First batch normalization layer.
        conv2: Second convolutional layer.
        bn2: Second batch normalization layer.
        shortcut: Residual layer(s).
    """

    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
    ):
        """Initialize ResNet building block.

        Args:
            in_planes: Number of input filters for first convolutional layer.
            planes: Number of output filters for first convolutional layer.
            stride: Stride for first convolutiona layer.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            """
            For CIFAR10 ResNet paper uses option A.
            "The shortcut still performs identity mapping, with extra zero entries
            padded for increasing dimensions. This option introduces no extra
            parameter."
            """
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=0.5, mode="nearest"),
                nn.ConstantPad3d((0, 0, 0, 0, planes // 4, planes // 4), 0),
            )

    def forward(self, x):
        """Forward pass."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """Residual Network for CIFAR10/100 datasets.

    Attributes:
        in_planes: Number of filters for first convolutional layer of ResNet blocks.
        conv1: First convolutional layer.
        bn1: First batch normalization layer.
        layer1: First ResNet block group.
        layer2: Second ResNet block group.
        layer3: Third ResNet block group.
        linear: Final linear layer.
    """

    def __init__(self, num_blocks, block=BasicBlock, num_classes=10):
        """Initialize ResNet.

        Args:
            num_blocks: Number of blocks stacked for each ResNet block group.
            block: Class for ResNet block.
            num_classes: Number of classes to predict from.
        """
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, num_classes)

        # From pytorch/vision Commit 0d75d9e torchvision/models/resnet.py Line 208
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Initialize ResNet building block.

        Args:
            block: Class for ResNet block.
            planes: Number of output filters for first block.
            num_blocks: Number of blocks to create for this ResNet block group.
            stride: Stride to create each block.

        Returns:
            Group of ResNet blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
