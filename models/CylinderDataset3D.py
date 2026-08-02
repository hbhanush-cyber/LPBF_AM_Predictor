
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CylinderDataset3D(Dataset):

    def __init__(self, dataset, window=10, augment=False, max_shift=10, input_channels=None):
        self.window = window
        self.augment = augment
        self.max_shift = max_shift
        self.input_channels = input_channels
        self.X = []
        self.Y = []

        for image, label in zip(dataset["X"], dataset["Y"]):

            if self.input_channels is not None:
                image = image[self.input_channels]

            h, w = image.shape[1:]

            new_h = ((h + 15) // 16) * 16
            new_w = ((w + 15) // 16) * 16

            pad_h = new_h - h
            pad_w = new_w - w

            image = F.pad(image, (0, pad_w, 0, pad_h))
            label = F.pad(label, (0, pad_w, 0, pad_h))

            self.X.append(image.float())
            self.Y.append(label.float())

    def __len__(self):
        return len(self.X)

    def shift_tensor(self, x, dx, dy, fill=0.0):

        H = x.shape[-2]
        W = x.shape[-1]

        shifted = torch.full_like(x, fill)

        src_y0 = max(0, -dy)
        src_y1 = min(H, H - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(W, W - dx)

        dst_y0 = max(0, dy)
        dst_y1 = min(H, H + dy)
        dst_x0 = max(0, dx)
        dst_x1 = min(W, W + dx)

        shifted[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]

        return shifted

    def __getitem__(self, idx):

        start = max(idx - (self.window - 1), 0)

        indices = list(range(start, idx + 1))

        while len(indices) < self.window:
            indices.insert(0, indices[0])

        frames = [self.X[i] for i in indices]

        volume = torch.stack(frames, dim=1)

        label = self.Y[idx]

        if self.augment:

            dx = random.randint(-self.max_shift, self.max_shift)
            dy = random.randint(-self.max_shift, self.max_shift)

            volume = self.shift_tensor(volume, dx=dx, dy=dy, fill=0.0)
            label = self.shift_tensor(label, dx=dx, dy=dy, fill=0.0)

        return volume, label
