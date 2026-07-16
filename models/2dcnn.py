import h5py
import numpy as np
import cv2
from pathlib import Path

HDF5_FILE = r"C:\Users\hrida\Downloads\2024-05-01 M2 AMMTO Fatigue Blanks 05.hdf5"
IR0_PATH = "/slices/camera_data/nir/0"
IR1_PATH = "/slices/camera_data/nir/1"
IR2_PATH = "/slices/camera_data/nir/2"
VIS0_PATH = "/slices/camera_data/visible/0"
VIS1_PATH = "/slices/camera_data/visible/1"
XCT_PATH = "/slices/registered_data/x-ray_ct_flaw"


LAYER = 535


X_MIN = 2095
X_MAX = 2281
Y_MIN = 1897
Y_MAX = 2088


XCT_SCALE = 13752 / 2844


OUTPUT_FILE = rf"layer{LAYER}_cylinder_training.npz"


with h5py.File(HDF5_FILE, "r") as f:
    ir0 = f[IR0_PATH][LAYER]
    ir1 = f[IR1_PATH][LAYER]
    ir2 = f[IR2_PATH][LAYER]
    vis0 = f[VIS0_PATH][LAYER]
    vis1 = f[VIS1_PATH][LAYER]
    xct_full = f[XCT_PATH][LAYER]

print("ir0 shape (full):", ir0.shape)
print("xct shape (full):", xct_full.shape)
print("computed scale:", xct_full.shape[0] / ir0.shape[0], "vs expected", XCT_SCALE)


ir0_c = ir0[Y_MIN:Y_MAX, X_MIN:X_MAX]
ir1_c = ir1[Y_MIN:Y_MAX, X_MIN:X_MAX]
ir2_c = ir2[Y_MIN:Y_MAX, X_MIN:X_MAX]
vis0_c = vis0[Y_MIN:Y_MAX, X_MIN:X_MAX]
vis1_c = vis1[Y_MIN:Y_MAX, X_MIN:X_MAX]


xct_x_min = int(round(X_MIN * XCT_SCALE))
xct_x_max = int(round(X_MAX * XCT_SCALE))
xct_y_min = int(round(Y_MIN * XCT_SCALE))
xct_y_max = int(round(Y_MAX * XCT_SCALE))


xct_h, xct_w = xct_full.shape[:2]
xct_x_min = max(0, xct_x_min)
xct_y_min = max(0, xct_y_min)
xct_x_max = min(xct_w, xct_x_max)
xct_y_max = min(xct_h, xct_y_max)

xct_cropped = xct_full[xct_y_min:xct_y_max, xct_x_min:xct_x_max]

print("XCT crop bounds (XCT pixel space):",
      xct_x_min, xct_x_max, xct_y_min, xct_y_max)
print("XCT cropped shape (before resize):", xct_cropped.shape)


new_width = ir0_c.shape[1]
new_height = ir0_c.shape[0]

xct_resized = cv2.resize(
    xct_cropped,
    (new_width, new_height),
    interpolation=cv2.INTER_NEAREST
)

print("XCT resized shape (matches IR crop):", xct_resized.shape)


print("XCT value range:", xct_resized.min(), xct_resized.max())
if xct_resized.max() > 255:
    print("WARNING: XCT values exceed 255 - uint8 cast will truncate/wrap them!")


X = np.stack(
    [
        ir0_c,
        ir1_c,
        ir2_c,
        vis0_c,
        vis1_c,
    ],
    axis=-1
)
Y = xct_resized.astype(np.uint8)


np.savez_compressed(
    OUTPUT_FILE,
    X=X,
    Y=Y
)

print("Done.")
print("Input shape :", X.shape)
print("Target shape:", Y.shape)
print("Channels:", X.shape[-1])

# ============================================================
# VISUAL VERIFICATION - confirm crop + registration look correct
# ============================================================
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

axes[0].imshow(ir0, cmap="hot")
axes[0].set_title(f"IR0 FULL (uncropped) - shape {ir0.shape}")
axes[0].axvline(X_MIN, color="cyan")
axes[0].axvline(X_MAX, color="cyan")
axes[0].axhline(Y_MIN, color="cyan")
axes[0].axhline(Y_MAX, color="cyan")

axes[1].imshow(ir0_c, cmap="hot")
axes[1].set_title(f"IR0 CROPPED - shape {ir0_c.shape}")

axes[2].imshow(Y, cmap="gray")
axes[2].set_title(f"XCT CROPPED+RESIZED - shape {Y.shape}")

axes[3].imshow(ir0_c, cmap="hot", alpha=0.6)
axes[3].imshow(Y, cmap="cool", alpha=0.5)
axes[3].set_title("OVERLAY - IR + XCT defects")

plt.tight_layout()
plt.savefig(f"verify_layer{LAYER}.png", dpi=150)
plt.close()
print(f"Saved verify_layer{LAYER}.png - open this to check alignment")

