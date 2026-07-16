import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # non-interactive backend - guarantees no popup, just saves to file
import matplotlib.pyplot as plt
import time
# ============================================================
# MODEL DEFINITION - must match the architecture used to train
# the checkpoint being loaded, exactly, or load_state_dict will fail
# ============================================================
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


# ============================================================
# USER SETTINGS
# ============================================================
MODEL_CHECKPOINT = r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\data\final_model_e350CombingData.pt"
DATA_FILE = r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\layers525-650CYLINDER48.pt"
THRESHOLD = 0.60
NUM_RANDOM_SAMPLES = 6
PLOT_OUTPUT_NAME = "test_results_random_samples.png"


# ============================================================
# LOAD MODEL
# ============================================================
t0 = time.time()
model = CNN()
model.load_state_dict(torch.load(MODEL_CHECKPOINT))
model.eval()
print(f"Model loaded ({time.time()-t0:.2f} sec)")

print(f"Testing on: {DATA_FILE}")
# ============================================================
# LOAD DATA
# ============================================================
t0 = time.time()
data = torch.load(DATA_FILE)
X_list = data["X"]
Y_list = data["Y"]
print(f"Data loaded: {len(X_list)} layers ({time.time()-t0:.2f} sec)")


# ============================================================
# RUN INFERENCE ONCE PER LAYER, COMPUTE METRICS
# ============================================================
t0 = time.time()

all_precision, all_recall, all_f1 = [], [], []
random_indices = random.sample(range(len(X_list)), min(NUM_RANDOM_SAMPLES, len(X_list)))
plot_data = {}

with torch.no_grad():
    for i, (X, Y) in enumerate(zip(X_list, Y_list)):
        pred = torch.sigmoid(model(X.unsqueeze(0)))[0, 0].numpy()
        groundTruth = Y[0].numpy()
        binaryPrediction = (pred > THRESHOLD).astype(np.float32)

        intersection = np.logical_and(binaryPrediction, groundTruth).sum()
        precision = intersection / (binaryPrediction.sum() + 1e-8)
        recall = intersection / (groundTruth.sum() + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

        if i in random_indices:
            plot_data[i] = (groundTruth, pred, binaryPrediction)

print(f"Inference + metrics: {time.time()-t0:.2f} sec")

print(f"\nMean Precision: {np.mean(all_precision):.4f}")
print(f"Mean Recall:    {np.mean(all_recall):.4f}")
print(f"Mean F1:        {np.mean(all_f1):.4f}")


# ============================================================
# PLOT RANDOM SAMPLES
# ============================================================
t0 = time.time()

fig, axes = plt.subplots(3, len(random_indices), figsize=(5*len(random_indices), 15))

for col, idx in enumerate(random_indices):
    gt, pred, bp = plot_data[idx]

    axes[0, col].imshow(gt, cmap="gray")
    axes[0, col].set_title(f"GT (idx {idx})")
    axes[0, col].axis("off")

    axes[1, col].imshow(pred, cmap="viridis")
    axes[1, col].set_title("Raw Prediction")
    axes[1, col].axis("off")

    axes[2, col].imshow(bp, cmap="gray")
    axes[2, col].set_title(f"Binary t={THRESHOLD}")
    axes[2, col].axis("off")

plt.tight_layout()
plt.savefig(PLOT_OUTPUT_NAME, dpi=150)
plt.close()

print(f"Plotting: {time.time()-t0:.2f} sec")
print(f"Saved: {PLOT_OUTPUT_NAME}")