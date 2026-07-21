import torch
import torch.nn as nn
import torch.nn.functional as F


class Convs3D(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv3d(inChannels, outChannels, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(outChannels)
        self.conv2 = nn.Conv3d(outChannels, outChannels, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(outChannels)

    def forward(self, image):
        image = F.relu(self.bn1(self.conv1(image)))
        image = F.relu(self.bn2(self.conv2(image)))
        return image


class encoder3D(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv = Convs3D(inChannels, outChannels)
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, image):
        down = self.conv(image)
        p = self.pool(down)
        return down, p


class decoder3D(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.up = nn.ConvTranspose3d(inChannels, outChannels, 2, 2)
        self.conv = Convs3D(outChannels * 2, outChannels)

    def forward(self, image, connection):
        image = self.up(image)

        if image.shape[-3:] != connection.shape[-3:]:
            image = F.interpolate(
                image,
                size=connection.shape[-3:],
                mode="trilinear",
                align_corners=False
            )

        image = torch.cat([image, connection], dim=1)

        return self.conv(image)


class uNet3D(nn.Module):
    def __init__(self, inChannels, numClasses):
        super().__init__()
        self.downConv1 = encoder3D(inChannels, 16)
        self.downConv2 = encoder3D(16, 32)
        self.downConv3 = encoder3D(32, 64)
        self.downConv4 = encoder3D(64, 128)

        self.bottleNeck = Convs3D(128, 256)

        self.upConv1 = decoder3D(256, 128)
        self.upConv2 = decoder3D(128, 64)
        self.upConv3 = decoder3D(64, 32)
        self.upConv4 = decoder3D(32, 16)

        self.out = nn.Conv3d(16, numClasses, 1)

    def forward(self, image):
        down1, p1 = self.downConv1(image)
        down2, p2 = self.downConv2(p1)
        down3, p3 = self.downConv3(p2)
        down4, p4 = self.downConv4(p3)

        bottleNeck = self.bottleNeck(p4)

        up1 = self.upConv1(bottleNeck, down4)
        up2 = self.upConv2(up1, down3)
        up3 = self.upConv3(up2, down2)
        up4 = self.upConv4(up3, down1)

        out = self.out(up4)
        return out