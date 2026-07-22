import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import time
import h5py
import cv2
from pathlib import Path
import random
from CNNUNET3D import uNet3D
from CylinderDataset3D import CylinderDataset3D

from uNET import Convs, encoder, decoder, uNet
from basicCNN import CNN
from cylinderDataSet import CylinderDataset

matplotlib.use('Agg')


class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight=None, dice_weight=1.5, bce_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice_score = (2. * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1 - dice_score.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


if Path("/content").exists():
    DATA_DIR = Path("/content/LBPF_ML_AM/models")
else:
    DATA_DIR = Path(r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models")


trainingDataFile = DATA_DIR / "layers525-650CYLINDER24.pt"
trainingDataFile1 = DATA_DIR / "layers525-650CYLINDER48.pt"
trainingDataFile2 = DATA_DIR / "layers525-650CYLINDER40.pt"
trainingDataFile3 = DATA_DIR / "layers525-650CYLINDER8.pt"
testingDataFile = DATA_DIR / "newTestNoV2real.pt"


dataset = CylinderDataset3D(
    torch.load(trainingDataFile, map_location="cpu")
)

dataset1 = CylinderDataset3D(
    torch.load(trainingDataFile1, map_location="cpu")
)

dataset2 = CylinderDataset3D(
    torch.load(trainingDataFile2, map_location="cpu")
)

dataset3 = CylinderDataset3D(
    torch.load(trainingDataFile3, map_location="cpu")
)


testData = CylinderDataset3D(
    torch.load(testingDataFile, map_location="cpu")
)


train_loader24 = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

train_loader40 = DataLoader(
    dataset2,
    batch_size=16,
    shuffle=True
)

train_loader8 = DataLoader(
    dataset3,
    batch_size=16,
    shuffle=True
)


train_loader48 = DataLoader(
    dataset1,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    testData,
    batch_size=16,
    shuffle=False
)


print(len(dataset))
print(len(dataset1))
print(len(dataset2))
print(len(testData))

print(len(train_loader24))
print(len(train_loader48))
print(len(train_loader40))
print(len(test_loader))


model = uNet3D(4, 1).to(device)

pos_weight = torch.tensor([30.0], device=device)
crit = DiceBCELoss(pos_weight=pos_weight, dice_weight=1.0, bce_weight=1.0)

optim = torch.optim.Adam(model.parameters(),lr=0.0001)

startTime = time.time()

e = 300

trainLoss = []
testLoss = []
trainCorrect = []
testCorrect = []


for i in range(e):

    epochStart = time.time()

    print(f"\nStarting Epoch {i + 1}/{e}")

    trainC = 0

    for loader in [train_loader24, train_loader48, train_loader40, train_loader8]:

        for b, (images, labels) in enumerate(loader):

            images = images.to(device)
            labels = labels.to(device)

            t0 = time.time()

            labelPred = model(images)

            loss = crit(
                labelPred,
                labels
            )

            t1 = time.time()

            pred = (
                torch.sigmoid(labelPred) > 0.5
            ).float()

            batchCorr = (
                pred == labels
            ).sum()

            trainC += batchCorr.item()

            optim.zero_grad()

            loss.backward()

            t2 = time.time()

            optim.step()

            t3 = time.time()

            if (b + 1) % 10 == 0:

                print(
                    f"Epoch {i + 1}/{e} "
                    f"Loader Batch {b + 1}/{len(loader)} "
                    f"Loss: {loss.item():.6f} "
                    f"Forward: {t1 - t0:.2f}s "
                    f"Backward: {t2 - t1:.2f}s "
                    f"Step: {t3 - t2:.2f}s"
                )

        trainLoss.append(loss.item())
        trainCorrect.append(trainC)

    epochTime = time.time() - epochStart

    print(
        f"Epoch {i + 1} completed "
        f"in {epochTime:.2f}s"
    )


    if (i + 1) % 50 == 0:

        torch.save(
            model.state_dict(),
            f"checkpoint_epoch{i + 1}.pt"
        )

        print(
            f"Saved checkpoint at epoch {i + 1}"
        )


testC = 0

model.eval()

with torch.no_grad():

    for b, (testImages, testLabels) in enumerate(test_loader):

        testImages = testImages.to(device)
        testLabels = testLabels.to(device)

        labelVal = model(testImages)

        predTest = (
            torch.sigmoid(labelVal) > 0.5
        ).float()

        testC += (
            predTest == testLabels
        ).sum().item()

        lossTest = crit(
            labelVal,
            testLabels
        )

        testLoss.append(
            lossTest.item()
        )

        testCorrect.append(
            testC
        )


currentTime = time.time()

totalTime = currentTime - startTime

print(
    f"Time was: {totalTime:.2f} seconds"
)


torch.save(
    model.state_dict(),
    "final_model_e300First3dCNNUnet10layerWindow.pt"
)

print(
    "Saved final model"
)


with torch.no_grad():

    testImages, testLabels = next(
        iter(test_loader)
    )

    testImages = testImages.to(device)

    prediction = model(testImages)

    prediction = torch.sigmoid(
        prediction
    )


testImages = testImages.cpu()
testLabels = testLabels.cpu()

# testImages shape: (B, C=4, D=5, H, W) -- take the current layer (last in depth window)
ir1 = testImages[0, 0, -1].numpy()
ir2 = testImages[0, 1, -1].numpy()
ir3 = testImages[0, 2, -1].numpy()
v1 = testImages[0, 3, -1].numpy()

groundTruth = testLabels[0, 0].numpy()  # unchanged: label is still (B, 1, H, W)

prediction = prediction.cpu()

prediction = prediction[0, 0].numpy()  # unchanged: model output squeezed back to (B, 1, H, W)

with torch.no_grad():

    for col, idx in enumerate(randomIndices):

        image, label = dataset[idx]

        imageBatch = image.unsqueeze(0).to(device)  # (1, C, D, H, W) -- unchanged, correct for 3D model input

        pred = torch.sigmoid(
            model(imageBatch)
        )

        binPred = (
            pred > THRESHOLD
        ).float()

        gt = label[0].numpy()  # unchanged: label is (1, H, W)

        predRaw = (
            pred[0, 0]
            .cpu()
            .numpy()
        )  # unchanged: model output already squeezed to (B, 1, H, W)

        predBinary = (
            binPred[0, 0]
            .cpu()
            .numpy()
        )