import torch
import torch.nn as nn
import math

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
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

        self.conv1_1_pan = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_2_pan = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=2)
        self.conv2_1_pan = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_2_pan = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2)

        self.conv1_lr = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2_lr = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.relu = nn.Relu()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n=m.kernel_size[0] * m.kernel_size[1] * m.out_channels
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
        out = self.relu(out)
        return out
