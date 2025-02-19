import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBlock, self).__init__()

        inner_padding = self._calculate_padding(kernel_size=kernel_size)

        self.batch_norm = nn.BatchNorm2d(num_features=in_channels)
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=2 * in_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=inner_padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=2 * in_channels,
                      out_channels=2 * in_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=inner_padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=2 * in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block(x)
        return x
    
    def _calculate_padding(self, kernel_size):
        padding = (kernel_size - 1) // 2
        return padding


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(UpConvBlock, self).__init__()

        padding = self._calculate_padding(kernel_size=kernel_size)

        self.batch_norm = nn.BatchNorm2d(num_features=2 * in_channels)
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=2 * in_channels, # because of concat
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block(x)
        return x
    
    def _calculate_padding(self, kernel_size):
        padding = (kernel_size - 1) // 2
        return padding


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_channels=8):
        super(UNet, self).__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=init_channels,
                      kernel_size=3,
                      padding=1),
            nn.ReLU()
        )

        self.conv_1 = ConvBlock(in_channels=init_channels,
                                out_channels=init_channels)
        self.conv_2 = ConvBlock(in_channels=init_channels,
                                out_channels=2 * init_channels)
        self.conv_3 = ConvBlock(in_channels=2 * init_channels,
                                out_channels=4 * init_channels)
        self.bottleneck = ConvBlock(in_channels=4 * init_channels,
                                    out_channels=4 * init_channels,
                                    padding=1)
        self.up_conv_1 = UpConvBlock(in_channels=4 * init_channels,
                                     out_channels=2 * init_channels)
        self.up_conv_2 = UpConvBlock(in_channels=2 * init_channels,
                                     out_channels=init_channels)
        self.up_conv_3 = UpConvBlock(in_channels=init_channels,
                                     out_channels=init_channels)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=init_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.input_conv(x)

        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)

        b = self.bottleneck(x3)

        x4 = torch.cat([x3, b], dim=1)
        x4 = self.up_conv_1(x4)

        x5 = torch.cat([x2, x4], dim=1)
        x5 = self.up_conv_2(x5)

        x6 = torch.cat([x1, x5], dim=1)
        x6 = self.up_conv_3(x6)

        out = self.out_conv(x6)
        return out



if __name__ == "__main__":
    model = UNet()
    sample_input = torch.randn(1, 3, 512, 512)
    output = model(sample_input)
    print(output.shape)
