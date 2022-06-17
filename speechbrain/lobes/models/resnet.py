# (modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

import torch
import torch.nn as nn
from speechbrain.nnet.CNN import Conv2d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import BatchNorm2d
from speechbrain.nnet.pooling import Pooling2d, AdaptivePool


class ResNet(nn.Module):

    def __init__(self, block_type, in_channels, layer_list, num_feats, num_classes, norm_layer=None, group=1,
                 width_per_group=64):
        """

        Parameters
        ----------
        block
        layer_list: (num_blocks, channels, stride)
        num_classes
        group
        width_per_group


        >>> resnet = ResNet(block='BasicBlock', in_channels=4,
        >>>    layer_list=[(2, 64, 2), (2, 128, 2), (2, 256, 2)], num_classes=10)
        """

        super(ResNet, self).__init__()
        if block_type.lower == 'bottleNeck':
            block = Bottleneck
        else:
            block = BasicBlock
        # layers = [Conv2d(in_channels=in_channels, out_channels=64, stride=2, kernel_size=3), Pooling2d('avg', 2)]
        cnn_layers = []

        last_out_channel = in_channels
        last_out_feats = num_feats
        for layer in layer_list:
            if len(layer) != 3:
                raise ValueError(
                    'Invalid input of layers, layer tuple should be: (num_blocks, output_channels, stride)')
            cnn_layers.append(ResNet_layer(block,
                                           num_blocks=layer[0],
                                           in_channels=last_out_channel,
                                           channels=layer[1],
                                           stride=layer[2],
                                           norm_layer=norm_layer,
                                           dilate=False,
                                           groups=group,
                                           width_per_group=width_per_group))
            last_out_feats = int((last_out_feats - 3 + 2) / layer[2]) + 1
            last_out_channel = layer[1]

        self.cnn_layers = nn.Sequential(*cnn_layers)
        linear_layers = [torch.nn.Linear(last_out_feats * last_out_channel, 250), torch.nn.Linear(250, num_classes)]
        self.linear_layers = nn.Sequential(*linear_layers)

    def forward(self, x):
        out = self.cnn_layers(x)
        out = torch.mean(out, dim=1)
        out = torch.flatten(out, start_dim=1)
        out = self.linear_layers(out)
        return out


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
        blocks = []
        if norm_layer is None:
            norm_layer = BatchNorm2d
        dilation = 1
        if dilate:
            dilation *= stride
            stride = 1
        blocks.append(block(in_channels, channels, stride, norm_layer,
                            groups, width_per_group))

        for _ in range(1, num_blocks):
            blocks.append(block(channels * block.expansion, channels, 1, norm_layer,
                                groups, width_per_group, dilation))

        super(ResNet_layer, self).__init__(*blocks)


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

    def __init__(self, in_channels, channels, stride=1, norm_layer=None,
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


# if __name__ == '__main__':
#     resnet = ResNet(block_type='BasicBlock', in_channels=1,
#                     layer_list=[(2, 64, 2), (2, 128, 1), (2, 256, 1)], num_feats=40, num_classes=10)
#     input_tensor = torch.rand(10, 15, 40)
#     input_tensor = torch.unsqueeze(input_tensor, dim=3)
#     print(input_tensor.shape)
#     out_tensor = resnet(input_tensor)
#     print(out_tensor.shape)
