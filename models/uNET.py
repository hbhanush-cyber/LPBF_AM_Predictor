import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid






class Convs(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(outChannels)


    def forward(self, image):
        image = F.relu(self.bn1(self.conv1(image)))
        image = F.relu(self.bn2(self.conv2(image)))

        return image


class encoder(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv = Convs(inChannels, outChannels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, image):
        down = self.conv(image)
        p = self.pool(down)
        return down, p


class decoder(nn.Module):
    def __init__(self,inChannels, outChannels):
        super().__init__()
        self.up = nn.ConvTranspose2d(inChannels, outChannels, 2, 2)
        self.conv = Convs(outChannels * 2, outChannels)

    def forward(self, image, connection):
        image = self.up(image)

        if image.shape[-2:] != connection.shape[-2:]:
            image = F.interpolate(
                image,
                size=connection.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        image = torch.cat([image, connection], dim=1)

        return self.conv(image)

class uNet(nn.Module):
    def __init__(self,inChannels, numClasses):
        super().__init__()
        self.downConv1 = encoder(inChannels,64)
        self.downConv2 = encoder(64, 128)
        self.downConv3 = encoder(128, 256)
        self.downConv4 = encoder(256, 512)

        self.bottleNeck = Convs(512,1024)

        self.upConv1 = decoder(1024,512)
        self.upConv2 = decoder(512,256)
        self.upConv3 = decoder(256,128)
        self.upConv4 = decoder(128,64)

        self.out = nn.Conv2d(64, numClasses,1)

    def forward(self,image):
        down1, p1 = self.downConv1(image)
        down2, p2 = self.downConv2(p1)
        down3, p3 = self.downConv3(p2)
        down4, p4 = self.downConv4(p3)

        bottleNeck = self.bottleNeck(p4)

        up1 = self.upConv1(bottleNeck,down4)
        up2 = self.upConv2(up1,down3)
        up3 = self.upConv3(up2,down2)
        up4 = self.upConv4(up3,down1)

        out = self.out(up4)
        return out

