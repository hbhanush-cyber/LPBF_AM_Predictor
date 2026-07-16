import torch
import numpy as np
from pyevtk.hl import imageToVTK

data = torch.load(r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\layers450-600.pt")
X_list = data["X"]
Y_list = data["Y"]

num_layers = len(X_list)
num_channels = X_list[0].shape[0]
H= X_list[0].shape[1]
W =X_list[0].shape[2]

print(f"Layers: {num_layers}, Channels: {num_channels}, Size: {H}x{W}")

channel_names = ["ir0", "ir1", "ir2", "vis0"]

volumes = {}
for c in range(num_channels):
    vol = np.stack([X[c].numpy() for X in X_list], axis=-1)
    name = channel_names[c] if c < len(channel_names) else f"channel_{c}"
    volumes[name] = vol.astype(np.float32).copy()

xct_volume = np.stack([Y[0].numpy() for Y in Y_list], axis=-1)  # (H, W, num_layers)
volumes["xct_flaws"] = xct_volume.astype(np.float32).copy()

imageToVTK(
    "cylinder_volume_450-600",
    pointData=volumes,
    spacing=(1.0, 1.0, 1.0)
)

print("Saved: cylinder_volume_550_580.vti")