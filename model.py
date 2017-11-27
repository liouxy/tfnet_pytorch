import torch
import torch.nn as nn
import math


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        identity_data = x
        output = self.prelu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.prelu = nn.PReLU()
        self.conv1_1_pan = nn.Conv2d(in_channels=1,
                                     out_channels=16,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.conv1_2_pan = nn.Conv2d(in_channels=16,
                                     out_channels=16,
                                     kernel_size=2,
                                     stride=2)
        self.conv2_1_pan = nn.Conv2d(in_channels=16,
                                     out_channels=32,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.conv2_2_pan = nn.Conv2d(in_channels=32,
                                     out_channels=32,
                                     kernel_size=2,
                                     stride=2)

        self.conv1_lr = nn.Conv2d(in_channels=4,
                                  out_channels=16,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.conv2_lr = nn.Conv2d(in_channels=16,
                                  out_channels=32,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64,
                                  out_channels=64,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.bn_mid = nn.BatchNorm2d(64)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        self.conv_output = nn.Conv2d(in_channels=64,
                                     out_channels=4,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x_pan, x_lr):
        out_pan = self.prelu(self.conv1_1_pan(x_pan))
        out_pan = self.prelu(self.conv1_2_pan(out_pan))
        out_pan = self.prelu(self.conv2_1_pan(out_pan))
        out_pan = self.prelu(self.conv2_2_pan(out_pan))

        out_lr = self.prelu(self.conv1_lr(x_lr))
        out_lr = self.prelu(self.conv2_lr(out_lr))

        out = torch.cat((out_lr, out_pan), dim=1)
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        out = self.tanh(out)
        return out


class TFNet(nn.Module):
    def __init__(self):
        super(TFNet, self).__init__()
        self.encoder1_pan=nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )
        self.encoder2_pan = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.encoder1_lr=nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU())
        self.encoder2_lr = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=2),
            nn.PReLU()
        )
        self.fusion1=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )
        self.fusion2=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=2,
                      stride=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=1,
                      stride=1),
            nn.PReLU()
        )
        self.restore1=nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128*4,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PixelShuffle(2),
            nn.PReLU())
        self.restore2=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64*4,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PixelShuffle(2),
            nn.PReLU())
        self.restore3=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=4,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_pan, x_lr):
        encoder1_pan = self.encoder1_pan(x_pan)
        encoder1_lr = self.encoder1_lr(x_lr)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        fusion1 = self.fusion1(torch.cat((encoder2_pan, encoder2_lr), dim=1))
        fusion2 = self.fusion2(fusion1)

        restore1 = self.restore1(fusion2)
        restore2 = self.restore2(torch.cat((restore1, fusion1),dim=1))

        restore3 = self.restore3(torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))

        return restore3





