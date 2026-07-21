import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CylinderDataset3D(Dataset):

    def __init__(self, dataset, window=10):

        self.window = window
        self.X = []
        self.Y = []

        # ==========================================
        # PAD ALL 2D LAYERS
        # ==========================================

        for image, label in zip(dataset["X"], dataset["Y"]):

            # image: (C, H, W)
            # label: (1, H, W)

            h, w = image.shape[1:]

            # Make H and W divisible by 16
            new_h = ((h + 15) // 16) * 16
            new_w = ((w + 15) // 16) * 16

            pad_h = new_h - h
            pad_w = new_w - w

            # Pad spatial dimensions
            image = F.pad(
                image,
                (0, pad_w, 0, pad_h)
            )

            label = F.pad(
                label,
                (0, pad_w, 0, pad_h)
            )

            self.X.append(image)
            self.Y.append(label)


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):

        # ==========================================
        # GET WINDOW OF LAYERS
        # ==========================================

        start = max(
            idx - (self.window - 1),
            0
        )

        indices = list(
            range(start, idx + 1)
        )


        # ==========================================
        # PAD BEGINNING OF BUILD
        # ==========================================

        # If we don't have enough previous layers,
        # repeat the earliest available layer.

        while len(indices) < self.window:

            indices.insert(
                0,
                indices[0]
            )


        # ==========================================
        # CREATE 3D INPUT
        # ==========================================

        # Each frame:
        # (C, H, W)

        frames = [
            self.X[i]
            for i in indices
        ]

        # Stack along depth dimension
        #
        # Before:
        # window × (C, H, W)
        #
        # After:
        # (C, D, H, W)

        volume = torch.stack(
            frames,
            dim=1
        )


        # ==========================================
        # CREATE 3D LABEL
        # ==========================================

        # Get the defect maps for the same
        # layers used in the input window.

        label_frames = [
            self.Y[i]
            for i in indices
        ]

        # Each label:
        # (1, H, W)
        #
        # Stack along depth dimension:
        #
        # (1, D, H, W)

        label_volume = torch.stack(
            label_frames,
            dim=1
        )


        return volume, label_volume