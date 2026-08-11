import torch
import torch.nn as nn
import torch.nn.functional as F


class Convs3D(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()

        self.conv1 = nn.Conv3d(inChannels, outChannels, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.GroupNorm(max(1, outChannels // 8),outChannels)

        self.conv2 = nn.Conv3d(outChannels, outChannels, kernel_size=3, stride=1, padding=1)

        self.bn2 = nn.GroupNorm(max(1, outChannels // 8),outChannels)
    

    def forward(self, image):
        image = F.relu(self.bn1(self.conv1(image)))

        image = F.relu(self.bn2(self.conv2(image)))

        return image


class encoder3D(nn.Module):

    def __init__(self, inChannels, outChannels):
        super().__init__()

        self.conv = Convs3D(inChannels, outChannels)


        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, image):
        down = self.conv(image)

        p = self.pool(down)

        return down, p


class decoder3D(nn.Module):

    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.up = nn.ConvTranspose3d(inChannels, outChannels, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv = Convs3D(outChannels * 2, outChannels)

    def forward(self, image, connection):
        image = self.up(image)

        if image.shape[-3:] != connection.shape[-3:]:
            image = F.interpolate(image, size=connection.shape[-3:], mode="trilinear", align_corners=False)

        image = torch.cat([image, connection], dim=1)

        return self.conv(image)


class uNet3D(nn.Module):

    def __init__(self, inChannels, numClasses, depth=10):
        super().__init__()

        self.downConv1 = encoder3D(inChannels, 16)

        self.downConv2 = encoder3D(16, 32)

        self.downConv3 = encoder3D(32, 64)

        self.bottleNeck = Convs3D(64, 128)

        self.upConv1 = decoder3D(128, 64)

        self.upConv2 = decoder3D(64, 32)

        self.upConv3 = decoder3D(32, 16)


        self.depthFuse = nn.Conv3d(16, 16, kernel_size=(depth, 1, 1))

        self.out = nn.Conv3d(16, numClasses, kernel_size=1)

    def forward(self, image):
        down1, p1 = self.downConv1(image)

        down2, p2 = self.downConv2(p1)

        down3, p3 = self.downConv3(p2)

        bottleNeck = self.bottleNeck(p3)

        up1 = self.upConv1(bottleNeck, down3)

        up2 = self.upConv2(up1, down2)

        up3 = self.upConv3(up2, down1)


        fused = self.depthFuse(up3)


        out = self.out(fused)

        return out.squeeze(2)
