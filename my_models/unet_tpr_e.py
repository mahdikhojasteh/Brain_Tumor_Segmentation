import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.doubleConv = nn.Sequential(
            nn.BatchNorm2d(channel),
            # nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            # nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )
        self.BnPlusRelu = nn.Sequential(
            nn.BatchNorm2d(channel),
            # nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        fx = self.doubleConv(x) + x
        h = self.BnPlusRelu(fx)
        return h


class TPR_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        out_channel = out_channel//2
        self.right_branch = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            ResidualBlock(out_channel)
        )
        self.left_branch = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 5, 1, 2),
            nn.BatchNorm2d(out_channel),
            # nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.right_branch(x)
        x2 = self.left_branch(x)
        fx = torch.cat((x1, x2), dim=1)
        return fx
    
class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dc2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, stride=1, padding=1),
            # nn.InstanceNorm2d(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                    kernel_size=3, stride=1, padding=1),
            # nn.InstanceNorm2d(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.dc2d(x)
   


def conv2DTransposet(in_channel, out_channel):
    return nn.ConvTranspose2d(in_channel, out_channel, 2, 2, 0)


class TPR_E_Unet(nn.Module):
    def __init__(self, in_channel=4, out_channel=4):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down1 = TPR_Block(in_channel, 16)
        self.down2 = TPR_Block(16, 32)
        self.down3 = TPR_Block(32, 64)

        self.bottle_neck = TPR_Block(64, 128)

        self.up1 = DoubleConv2d(128, 64)
        self.up2 = DoubleConv2d(64, 32)
        self.up3 = DoubleConv2d(32, 16)

        self.upsample1 = conv2DTransposet(128, 64)
        self.upsample2 = conv2DTransposet(64, 32)
        self.upsample3 = conv2DTransposet(32, 16)

        self.output = nn.Sequential(
            nn.Conv2d(16, out_channel, 1, 1, 0),
            # nn.Sigmoid()
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out1 = self.down1(x)
        out1pooled = self.max_pool(out1)
        out2 = self.down2(out1pooled)
        out2pooled = self.max_pool(out2)
        out3 = self.down3(out2pooled)
        out3pooled = self.max_pool(out3)

        out4 = self.bottle_neck(out3pooled)
        out4_transposed = self.upsample1(out4)

        concat1 = torch.cat((out4_transposed, out3), dim=1)

        out1m = self.up1(concat1)
        out1m_transposed = self.upsample2(out1m)

        concat2 = torch.cat((out1m_transposed, out2), dim=1)

        out2m = self.up2(concat2)
        out2m_transposed = self.upsample3(out2m)

        concat3 = torch.cat((out2m_transposed, out1), dim=1)

        out3m = self.up3(concat3)

        final = self.output(out3m)

        return final



if __name__ == "__main__":
    net = TPR_E_Unet(in_channel=4, out_channel=4)
    x = torch.randn((10, 4, 192, 192))
    y = net(x)
    print('pred.shape', y.shape)

