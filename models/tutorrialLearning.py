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
import numpy as np
from dataExtraction import Extractor
from uNET import Convs, encoder, decoder, uNet
from basicCNN import CNN
from cylinderDataSet import CylinderDataset
matplotlib.use('TkAgg')


X_MIN = 366
X_MAX = 545
Y_MIN = 1448
Y_MAX = 1667
XCT_SCALE = 13752 / 2844

HDF5_FILE = r"C:\Users\hrida\Downloads\2024-05-01 M2 AMMTO Fatigue Blanks 05.hdf5"
IR0_PATH = "/slices/camera_data/nir/0"
IR1_PATH = "/slices/camera_data/nir/1"
IR2_PATH = "/slices/camera_data/nir/2"
VIS0_PATH = "/slices/camera_data/visible/0"
##VIS1_PATH = "/slices/camera_data/visible/1"
XCT_PATH = "/slices/registered_data/x-ray_ct_flaw"

LAYER_MIN = 750
LAYER_MAX = 850



trainingDataFile = Path(r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\layers525-650CYLINDER24.pt")
trainingDataFile1 = Path(r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\layers525-650CYLINDER48.pt")
testingDataFile = Path(r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\newTestNoV2real.pt")
##if not trainingDataFile.exists():
##data = torch.load()
##trainExtractor = Extractor(xMin=X_MIN, xMax=X_MAX, yMin=Y_MIN, yMax=Y_MAX,layerMin=450,layerMax=600)
dataset = CylinderDataset(torch.load(trainingDataFile))
dataset1 = CylinderDataset(torch.load(trainingDataFile1))

##dataset = CylinderDataset(trainExtractor.extract("layers750-850CYLINDER2"))
##testExtractor = Extractor(xMin=X_MIN, xMax=X_MAX, yMin=Y_MIN, yMax=Y_MAX,layerMin=530,layerMax=531)
testData = CylinderDataset(torch.load(testingDataFile))
##testData = CylinderDataset(trainExtractor.extract("newTestNoV2real"))


train_loader24 = DataLoader(
    dataset,
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
print(len(testData))

print(len(train_loader24))
print(len(train_loader48))
print(len(test_loader))

model = uNet(4, 1)
pos_weight = torch.tensor([30.0])
crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

startTime = time.time()

e = 5

trainLoss = []
testLoss = []
trainCorrect = []
testCorrect = []

for i in range(e):
    epochStart = time.time()
    print(f"\nStarting Epoch {i+1}/{e}")
    trainC = 0
    testC = 0
    for loader in [train_loader24, train_loader48]:
        if (i + 1) % 50 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch{i + 1}.pt")
            print(f"Saved checkpoint at epoch {i + 1}")

        for b, (images, labels) in enumerate(loader):
            t0 = time.time()

            b += 1
            labelPred = model(images)
            loss = crit(labelPred, labels)
            t1 = time.time()

            pred = (torch.sigmoid(labelPred) > 0.5).float()
            batchCorr = (pred == labels).sum()
            trainC = trainC + batchCorr

            optim.zero_grad()
            loss.backward()
            t2 = time.time()
            optim.step()
            print(
                f"Batch {b}: "
                f"forward={t1 - t0:.2f}s "
                f"backward={t2 - t1:.2f}s "
                f"step={t3 - t2:.2f}s"
            )
            if (i + 1) % 50 == 0:
                print(f"Epoch {i + 1}/{e}, Loss: {loss.item():.6f}")
        trainLoss.append(loss.item())
        trainCorrect.append(trainC)

with torch.no_grad():
     for b, (testImages, testLabels) in enumerate(test_loader):
        labelVal = model(testImages)
        predTest = (torch.sigmoid(labelVal) > 0.5).float()
        testC += (predTest == testLabels).sum()
        lossTest = crit(labelVal, testLabels)
        testLoss.append(lossTest.item())
        testCorrect.append(testC)

currentTime = time.time()
totalTime = currentTime - startTime
print(f'Time was : {totalTime}')

torch.save(model.state_dict(), "final_model_e350.pt")
print("Saved final model")

model.eval()

with torch.no_grad():
    testImages, testLabels = next(iter(test_loader))
    prediction = model(testImages)
    prediction = torch.sigmoid(prediction)

ir1 = testImages[0, 0].cpu().numpy()
ir2 = testImages[0, 1].cpu().numpy()
ir3 = testImages[0, 2].cpu().numpy()
v1 = testImages[0, 3].cpu().numpy()
groundTruth = testLabels[0, 0].cpu().numpy()
prediction = prediction[0, 0].cpu().numpy()

binaryPrediction = (prediction > 0.5).astype(float)

intersection = np.logical_and(binaryPrediction, groundTruth).sum()
union = np.logical_or(binaryPrediction, groundTruth).sum()

precision = intersection / (binaryPrediction.sum() + 1e-8)
recall = intersection / (groundTruth.sum() + 1e-8)
iou = intersection / (union + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
pixelAccuracy = (binaryPrediction == groundTruth).mean()

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
plt.imshow(v1, cmap="managua")
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
plt.text(0.1, 0.5, stats_text, fontsize=16, family="monospace", va="center")
plt.title("Test Metrics")

plt.tight_layout()
plt.savefig("prediction_overview.png", dpi=150)
plt.close()

print("im doneee")

import random

N = 6  # number of random layers to sample
THRESHOLD = 0.5
model.eval()
# pick N random indices from the full training set
totalTrain = len(dataset)
randomIndices = random.sample(range(totalTrain), N)
fig, axes = plt.subplots(3, N, figsize=(4 * N, 12))
with torch.no_grad():
    for col, idx in enumerate(randomIndices):
        image, label = dataset[idx]
        # add batch dimension for the model
        imageBatch = image.unsqueeze(0)
        pred = torch.sigmoid(model(imageBatch))
        binPred = (pred > THRESHOLD).float()
        gt = label[0].cpu().numpy()
        predRaw = pred[0, 0].cpu().numpy()
        predBinary = binPred[0, 0].cpu().numpy()
        axes[0, col].imshow(gt, cmap="gray")
        axes[0, col].set_title(f"GT (idx {idx})")
        axes[0, col].axis("off")
        axes[1, col].imshow(predRaw, cmap="viridis")
        axes[1, col].set_title("Raw Prediction")
        axes[1, col].axis("off")
        axes[2, col].imshow(predBinary, cmap="gray")
        axes[2, col].set_title(f"Binary t={THRESHOLD}")
        axes[2, col].axis("off")
plt.tight_layout()
plt.savefig(f"random_train_layers_t{THRESHOLD}.png", dpi=150)
plt.close()
print(f"Saved random_train_layers_t{THRESHOLD}.png")
print(f"Sampled indices: {randomIndices}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(trainLoss)
plt.title("Train Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(testLoss)
plt.title("Test Loss per Batch (across all epochs)")
plt.xlabel("Batch (cumulative)")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
plt.close()
print("Saved loss_curves.png")
