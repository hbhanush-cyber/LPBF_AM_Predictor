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

from uNET import Convs, encoder, decoder, uNet
from basicCNN import CNN
from cylinderDataSet import CylinderDataset

matplotlib.use('Agg')


class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight=None, dice_weight=1.0, bce_weight=1.0, smooth=1.0):
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
##chnage so i can commit

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
testingDataFile = DATA_DIR / "newTestNoV2real.pt"


dataset = CylinderDataset(
    torch.load(trainingDataFile, map_location="cpu")
)

dataset1 = CylinderDataset(
    torch.load(trainingDataFile1, map_location="cpu")
)

dataset2 = CylinderDataset(
    torch.load(trainingDataFile2, map_location="cpu")
)

testData = CylinderDataset(
    torch.load(testingDataFile, map_location="cpu")
)

#commet
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


model = uNet(4, 1).to(device)

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

    for loader in [train_loader24, train_loader48]:

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
    "final_model_e350.pt"
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

ir1 = testImages[0, 0].numpy()
ir2 = testImages[0, 1].numpy()
ir3 = testImages[0, 2].numpy()
v1 = testImages[0, 3].numpy()

groundTruth = testLabels[0, 0].numpy()

prediction = prediction.cpu()

prediction = prediction[0, 0].numpy()


binaryPrediction = (
    prediction > 0.5
).astype(float)


intersection = np.logical_and(
    binaryPrediction,
    groundTruth
).sum()

union = np.logical_or(
    binaryPrediction,
    groundTruth
).sum()


precision = intersection / (
    binaryPrediction.sum() + 1e-8
)

recall = intersection / (
    groundTruth.sum() + 1e-8
)

iou = intersection / (
    union + 1e-8
)

f1 = 2 * precision * recall / (
    precision + recall + 1e-8
)

pixelAccuracy = (
    binaryPrediction == groundTruth
).mean()


plt.figure(figsize=(24, 12))


plt.subplot(2, 4, 1)
plt.imshow(ir1, cmap="hot")
plt.title("IR Channel 0")
plt.colorbar()


plt.subplot(2, 4, 2)
plt.imshow(ir2, cmap="hot")
plt.title("IR Channel 1")
plt.colorbar()


plt.subplot(2, 4, 3)
plt.imshow(ir3, cmap="hot")
plt.title("IR Channel 2")
plt.colorbar()


plt.subplot(2, 4, 4)
plt.imshow(v1, cmap="magma")
plt.title("Visible Light 1")
plt.colorbar()


plt.subplot(2, 4, 5)
plt.imshow(groundTruth, cmap="gray")
plt.title("Ground Truth XCT")
plt.colorbar()


plt.subplot(2, 4, 6)
plt.imshow(binaryPrediction, cmap="gray")
plt.title("Binary Prediction")


plt.subplot(2, 4, 7)
plt.imshow(prediction, cmap="viridis")
plt.title("CNN Prediction")
plt.colorbar()


plt.subplot(2, 4, 8)
plt.axis("off")

stats_text = (
    f"Precision: {precision:.4f}\n"
    f"Recall:    {recall:.4f}\n"
    f"IoU:       {iou:.4f}\n"
    f"F1:        {f1:.4f}\n"
    f"Pixel Acc: {pixelAccuracy:.4f}"
)

plt.text(
    0.1,
    0.5,
    stats_text,
    fontsize=16,
    family="monospace",
    va="center"
)

plt.title("Test Metrics")

plt.tight_layout()

plt.savefig(
    "prediction_overview.png",
    dpi=150
)

plt.close()

print("im doneee")


N = 6

THRESHOLD = 0.5

model.eval()

totalTrain = len(dataset)

randomIndices = random.sample(
    range(totalTrain),
    N
)


fig, axes = plt.subplots(
    3,
    N,
    figsize=(4 * N, 12)
)


with torch.no_grad():

    for col, idx in enumerate(randomIndices):

        image, label = dataset[idx]

        imageBatch = image.unsqueeze(0).to(device)

        pred = torch.sigmoid(
            model(imageBatch)
        )

        binPred = (
            pred > THRESHOLD
        ).float()

        gt = label[0].numpy()

        predRaw = (
            pred[0, 0]
            .cpu()
            .numpy()
        )

        predBinary = (
            binPred[0, 0]
            .cpu()
            .numpy()
        )


        axes[0, col].imshow(
            gt,
            cmap="gray"
        )

        axes[0, col].set_title(
            f"GT (idx {idx})"
        )

        axes[0, col].axis("off")


        axes[1, col].imshow(
            predRaw,
            cmap="viridis"
        )

        axes[1, col].set_title(
            "Raw Prediction"
        )

        axes[1, col].axis("off")


        axes[2, col].imshow(
            predBinary,
            cmap="gray"
        )

        axes[2, col].set_title(
            f"Binary t={THRESHOLD}"
        )

        axes[2, col].axis("off")


plt.tight_layout()

plt.savefig(
    f"random_train_layers_t{THRESHOLD}.png",
    dpi=150
)

plt.close()


print(
    f"Saved random_train_layers_t{THRESHOLD}.png"
)

print(
    f"Sampled indices: {randomIndices}"
)


plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)

plt.plot(trainLoss)

plt.title(
    "Train Loss"
)

plt.xlabel(
    "Training Dataset"
)

plt.ylabel(
    "Loss"
)


plt.subplot(1, 2, 2)

plt.plot(testLoss)

plt.title(
    "Test Loss per Batch"
)

plt.xlabel(
    "Batch"
)

plt.ylabel(
    "Loss"
)


plt.tight_layout()

plt.savefig(
    "loss_curves.png"
)

plt.close()


print(
    "Saved loss_curves.png"
)