import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pyevtk.hl import imageToVTK
from torch.utils.data import Dataset
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import h5py
import cv2
from pathlib import Path
import numpy as np
from dataExtraction import Extractor

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 1, 1)

    def forward(self, image):
        image = F.relu(self.bn1(self.conv1(image)))
        image = F.relu(self.bn2(self.conv2(image)))
        image = F.relu(self.bn3(self.conv3(image)))
        image = F.relu(self.bn4(self.conv4(image)))
        image = self.conv5(image)
        return image


MODEL_CHECKPOINT = r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\final_model_e350.pt"
DATA_FILE = r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\layers525-650CYLINDER48.pt"
VTI_OUTPUT_NAME = "model_output_volumev2"  # produces model_output_volume.vti

PVSM_STATE_FILE = r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\Visualization.pvsm"  # <-- set your actual .pvsm path
VTI_FULL_PATH = r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\model_output_volume.vti"  # imageToVTK saves relative to cwd, adjust if needed

model = CNN()
model.load_state_dict(torch.load(MODEL_CHECKPOINT))
model.eval()
data = torch.load(DATA_FILE)
X_list = data["X"]
Y_list = data["Y"]
all_f1, all_precision, all_recall = [], [], []
channel_names = ["ir0", "ir1", "ir2", "vis0"]
volumes = {name: [] for name in channel_names}
volumes["xct_flaws"] = []
volumes["prediction"] = []
volumes["binaryPrediction"] = []

random_indices = random.sample(range(len(X_list)), min(6, len(X_list)))
plot_data = {}  # store just the 6 samples needed for the figure

with torch.no_grad():
    for i, (X, Y) in enumerate(zip(X_list, Y_list)):
        pred = torch.sigmoid(model(X.unsqueeze(0)))[0, 0].numpy()
        groundTruth = Y[0].numpy()
        binaryPrediction = (pred > 0.7).astype(np.float32)

        # metrics
        intersection = np.logical_and(binaryPrediction, groundTruth).sum()
        precision = intersection / (binaryPrediction.sum() + 1e-8)
        recall = intersection / (groundTruth.sum() + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

        # volume data
        for c, name in enumerate(channel_names):
            volumes[name].append(X[c].numpy())
        volumes["xct_flaws"].append(groundTruth)
        volumes["prediction"].append(pred)
        volumes["binaryPrediction"].append(binaryPrediction)

        if i in random_indices:
            plot_data[i] = (groundTruth, pred, binaryPrediction)

print(f"Mean Precision: {np.mean(all_precision):.4f}")
print(f"Mean Recall: {np.mean(all_recall):.4f}")
print(f"Mean F1: {np.mean(all_f1):.4f}")

# plotting
fig, axes = plt.subplots(3, 6, figsize=(30, 15))
for col, idx in enumerate(random_indices):
    gt, pred, bp = plot_data[idx]
    axes[0, col].imshow(gt, cmap="gray"); axes[0, col].set_title(f"GT (idx {idx})"); axes[0, col].axis("off")
    axes[1, col].imshow(pred, cmap="viridis"); axes[1, col].set_title("Raw Prediction"); axes[1, col].axis("off")
    axes[2, col].imshow(bp, cmap="gray"); axes[2, col].set_title("Binary t=0.7"); axes[2, col].axis("off")
plt.tight_layout()
plt.savefig("new_cylinder_random_samples.png", dpi=150)
plt.close()

# vti export
final_volumes = {name: np.stack(vol_list, axis=-1).astype(np.float32).copy() for name, vol_list in volumes.items()}
imageToVTK(VTI_OUTPUT_NAME, pointData=final_volumes, spacing=(1.0, 1.0, 1.0))
print(f"Saved: {VTI_OUTPUT_NAME}.vti")

