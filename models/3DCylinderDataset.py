import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CylinderDataset3D(Dataset):

    def __init__(self, dataset, window=10):
        self.window = window
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

            self.X.append(image)   # (C, H, W)
            self.Y.append(label)   # (1, H, W)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        start = max(idx - (self.window - 1), 0)
        indices = list(range(start, idx + 1))

        # pad the front by repeating the earliest available layer
        while len(indices) < self.window:
            indices.insert(0, indices[0])

        frames = [self.X[i] for i in indices]      # each (C, H, W)
        volume = torch.stack(frames, dim=1)         # -> (C, D=window, H, W)

        label = self.Y[idx]                          # (1, H, W) — current layer only

        return volume, label