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

class CylinderDataset(Dataset):

    def __init__(self, dataset):
        self.X = []
        self.Y = []

        for image, label in zip(dataset["X"], dataset["Y"]):

            h, w = image.shape[1:]

            new_h = ((h + 15) // 16) * 16
            new_w = ((w + 15) // 16) * 16

            pad_h = new_h - h
            pad_w = new_w - w

            image = F.pad(image, (0, pad_w, 0, pad_h))
            label = F.pad(label, (0, pad_w, 0, pad_h))

            self.X.append(image)
            self.Y.append(label)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]