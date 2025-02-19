import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dropout=False):
        super(ConvBlock, self).__init__()

        inner_padding = self._calculate_padding(kernel_size=kernel_size)

        self.batch_norm = nn.BatchNorm2d(num_features=in_channels)
        
        if not dropout:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                        out_channels=2 * in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=inner_padding),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=2 * in_channels,
                        out_channels=2 * in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=inner_padding),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=2 * in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=inner_padding),
                nn.LeakyReLU()
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                        out_channels=2 * in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=inner_padding),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=2 * in_channels,
                        out_channels=2 * in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=inner_padding),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=2 * in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=inner_padding),
                nn.LeakyReLU()
            )

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.block(x)
        return x
    
    def _calculate_padding(self, kernel_size):
        padding = (kernel_size - 1) // 2
        return padding


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dropout=False):
        super(UpConvBlock, self).__init__()

        inner_padding = self._calculate_padding(kernel_size=kernel_size)

        self.batch_norm = nn.BatchNorm2d(num_features=in_channels)
        
        if not dropout:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=inner_padding),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=inner_padding),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=2,
                                    stride=2),
                nn.LeakyReLU()
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, # because of concat
                            out_channels=in_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=inner_padding),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=inner_padding),
                nn.LeakyReLU(),
                nn.Dropout2d(p=0.2),
                nn.ConvTranspose2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=2,
                                    stride=2),
                nn.LeakyReLU()
            )


    def forward(self, x):
        x = self.batch_norm(x)
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
            nn.LeakyReLU()
        )

        self.conv_1 = ConvBlock(in_channels=init_channels,
                                out_channels=init_channels)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2 = ConvBlock(in_channels=init_channels,
                                out_channels=2 * init_channels)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3 = ConvBlock(in_channels=2 * init_channels,
                                out_channels=4 * init_channels,
                                dropout=True)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.up_conv_1 = UpConvBlock(in_channels=4 * init_channels,
                                    out_channels=4 * init_channels,
                                    dropout=True)
        
        self.up_conv_2 = UpConvBlock(in_channels=4 * init_channels,
                                     out_channels=2 * init_channels)
        self.up_conv_3 = UpConvBlock(in_channels=2 * init_channels,
                                     out_channels=init_channels)
        
        self.conv_4 = ConvBlock(in_channels=2 * init_channels,
                                out_channels=init_channels)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=init_channels,
                      out_channels=2,
                      kernel_size=3,
                      padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=2,
                        out_channels=out_channels,
                        kernel_size=1)
        )

    def forward(self, x):
        x = self.input_conv(x)

        x1_cat = self.conv_1(x)
        x1 = self.max_pool_1(x1_cat)

        x2_cat = self.conv_2(x1)
        x2 = self.max_pool_2(x2_cat)

        x3_cat = self.conv_3(x2)
        x3 = self.max_pool_3(x3_cat)

        b = self.up_conv_1(x3)

        x4 = torch.cat([b, x3_cat], dim=1)
        x4 = self.up_conv_2(b)
        
        x5 = torch.cat([x2_cat, x4], dim=1)
        x5 = self.up_conv_3(x4)
        
        x6 = torch.cat([x1_cat, x5], dim=1)

        x7 = self.conv_4(x6)
        out = self.out_conv(x7)

        return out



if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1, init_channels=8)
    sample_input = torch.randn(1, 1, 512, 512)
    output = model(sample_input)
    print(output.shape)
