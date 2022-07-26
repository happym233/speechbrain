import torch
import torch.nn as nn
import speechbrain
import torch.nn.functional as F


class Encoder(torch.nn.Module):

    def __init__(self, input_channel, output_channels_list, kernel_size=3, padding=1, stride=2):
        super(Encoder, self).__init__()
        conv_array = []
        prev = input_channel
        first = True
        for out_channels in output_channels_list:
            if not first:
                conv_array.append(nn.Sequential(
                    nn.Tanh(),
                    nn.BatchNorm1d(prev)
                ))
            else:
                first = False
            conv_array.append(nn.Sequential(
                nn.Conv1d(in_channels=prev,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          padding=padding,
                          stride=stride)
            ))

            prev = out_channels
        self.seq = nn.Sequential(*conv_array)

    def forward(self, x):
        return self.seq(x)


class Decoder(torch.nn.Module):

    def __init__(self, input_channel, output_channels_list, kernel_size=3, padding=1, stride=2):
        super(Decoder, self).__init__()
        conv_array = []
        prev = input_channel
        first = True
        for out_channels in output_channels_list:
            if not first:
                conv_array.append(nn.Sequential(
                    nn.Tanh(),
                    nn.BatchNorm1d(prev)
                ))
            else:
                first = False
            conv_array.append(nn.Sequential(
                nn.ConvTranspose1d(in_channels=prev,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   output_padding=padding,
                                   stride=stride),
            ))
            prev = out_channels
        self.seq = nn.Sequential(*conv_array)

    def forward(self, x):
        return self.seq(x)

if __name__ == '__main__':
    a = torch.rand([4, 1, 64])
    encoder = Encoder(1, [16, 8, 4])
    decoder = Decoder(4, [8, 16, 1])
    b = encoder(a)
    print(b.shape)
    c = decoder(b)
    print(c.shape)

