# (modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

import torch
import torch.nn as nn
from speechbrain.nnet.CNN import Conv2d
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import BatchNorm2d


class ResNet(Sequential):

    def __init__(self, block_list, num_classes):
        print('init')


class ResNet_layer(Sequential):

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 channels,
                 stride,
                 norm_layer=None,
                 dilate=False,
                 groups=1,
                 width_per_group=64):
        layers = []
        if norm_layer is None:
            norm_layer = BatchNorm2d
        dilation = 1
        if dilate:
            dilation *= stride
            stride = 1
        layers.append(block(in_channels, channels, stride, norm_layer,
                            groups, width_per_group))

        for _ in range(1, num_blocks):
            layers.append(block(channels * block.expansion, channels, 1, norm_layer,
                                groups, width_per_group, dilation))

        super(ResNet_layer, self).__init__(*layers)


def generate_downsample(in_channels, channels, expansion, stride, norm_layer):
    if stride == 1 and in_channels == channels * expansion:
        return None
    else:
        return nn.Sequential(Conv2d(in_channels=in_channels,
                                    out_channels=channels * expansion,
                                    kernel_size=1,
                                    stride=stride),
                             norm_layer(input_size=channels * expansion))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1,  norm_layer=None,
                 groups=1, base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=channels, stride=stride, kernel_size=3)
        self.bn1 = norm_layer(input_size=channels)
        self.conv2 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3)
        self.bn2 = norm_layer(input_size=channels)

        self.downsample = generate_downsample(in_channels, channels, self.expansion, stride, norm_layer)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)
        return out


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, norm_layer=None,
                 groups=1, base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        width = int(channels * (base_width / 64)) * groups
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=width, kernel_size=1)
        self.bn1 = norm_layer(input_size=width)
        self.conv2 = Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=stride, dilation=dilation)
        self.bn2 = norm_layer(input_size=width)
        self.conv3 = Conv2d(in_channels=width, out_channels=channels * self.expansion, kernel_size=1)
        self.bn3 = norm_layer(input_size=channels * self.expansion)
        self.activation = nn.ReLU(inplace=True)
        self.downsample = generate_downsample(in_channels, channels, self.expansion, stride, norm_layer)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)
        return out


if __name__ == '__main__':
    # basicBlock = BasicBlock(4, 4)
    # bottleNeck = Bottleneck(4, 4)
    inputs = torch.rand(10, 20, 20, 4)
    # outputs = basicBlock(inputs)
    # print(outputs.shape)
    # outputs = bottleNeck(inputs)
    # print(outputs.shape)
    resnet_layer = ResNet_layer(Bottleneck, 2, 4, 8, 2)
    outputs = resnet_layer(inputs)
    print(outputs.shape)
