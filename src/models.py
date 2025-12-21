import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorUNet(nn.Module):
    def __init__(self, input_shape=(3, 256, 256)):
        super(GeneratorUNet, self).__init__()
        channels = input_shape[0]

        # Downsampling block
        def down(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        # Upsampling block
        def up(in_feat, out_feat, dropout=0.0):
            layers = [
                nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return layers

        self.down1 = nn.Sequential(*down(channels, 64, normalize=False))
        self.down2 = nn.Sequential(*down(64, 128))
        self.down3 = nn.Sequential(*down(128, 256))
        self.down4 = nn.Sequential(*down(256, 512))
        self.down5 = nn.Sequential(*down(512, 512))
        self.down6 = nn.Sequential(*down(512, 512))
        self.down7 = nn.Sequential(*down(512, 512))
        self.down8 = nn.Sequential(*down(512, 512, normalize=False))

        self.up1 = nn.Sequential(*up(512, 512, dropout=0.5))
        self.up2 = nn.Sequential(*up(1024, 512, dropout=0.5))
        self.up3 = nn.Sequential(*up(1024, 512, dropout=0.5))
        self.up4 = nn.Sequential(*up(1024, 512))
        self.up5 = nn.Sequential(*up(1024, 256))
        self.up6 = nn.Sequential(*up(512, 128))
        self.up7 = nn.Sequential(*up(256, 64))

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8)
        u1 = torch.cat((u1, d7), 1)
        u2 = self.up2(u1)
        u2 = torch.cat((u2, d6), 1)
        u3 = self.up3(u2)
        u3 = torch.cat((u3, d5), 1)
        u4 = self.up4(u3)
        u4 = torch.cat((u4, d4), 1)
        u5 = self.up5(u4)
        u5 = torch.cat((u5, d3), 1)
        u6 = self.up6(u5)
        u6 = torch.cat((u6, d2), 1)
        u7 = self.up7(u6)
        u7 = torch.cat((u7, d1), 1)
        return self.final(u7)

class GeneratorResNet(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()
        
        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features

        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)