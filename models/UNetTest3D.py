import time
from pathlib import Path

import torch
import torch.nn as nn
from CNNUNET3D import uNet3D
from CylinderDataset3D import CylinderDataset3D
from torch.utils.data import DataLoader

# ==========================================================
# LOSS FUNCTION
# ==========================================================
x = 12


class DiceBCELoss(nn.Module):

    def __init__(self, pos_weight=None, dice_weight=1.0, bce_weight=1.0, smooth=1.0):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        # ------------------------------------------
        # BCE LOSS
        # ------------------------------------------

        bce_loss = self.bce(logits, targets)

        # ------------------------------------------
        # DICE LOSS
        # ------------------------------------------

        probs = torch.sigmoid(logits)

        probs_flat = probs.view(probs.size(0), -1)

        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (
                probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth)

        dice_loss = (1.0 - dice_score.mean())

        # ------------------------------------------
        # COMBINED LOSS
        # ------------------------------------------

        total_loss = (self.bce_weight * bce_loss + self.dice_weight * dice_loss)

        return total_loss


# ==========================================================
# DEVICE
# ==========================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: "
          f"{torch.cuda.get_device_name(0)}")

# ==========================================================
# SETTINGS
# ==========================================================

WINDOW = 5

BATCH_SIZE = 32

MAX_SHIFT = 80

EPOCHS = 800

LEARNING_RATE = 2e-4

NUM_WORKERS = 2

CHECKPOINT_INTERVAL = 25

# ==========================================================
# DATA DIRECTORY
# ==========================================================

if Path("/content").exists():

    DATA_DIR = Path("/content/LBPF_ML_AM/models")

else:

    DATA_DIR = Path(r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models")


training_files = [
    ("Cylinder2", DATA_DIR / "layers525-650CYLINDER2Updated.pt"),
    ("Cylinder3", DATA_DIR / "layers525-650CYLINDER3Updated.pt"),
    #("Cylinder4", DATA_DIR / "layers525-650CYLINDER4Updated.pt"),
    ("Cylinder5", DATA_DIR / "layers525-650CYLINDER5Updated.pt"),
    ("Cylinder6", DATA_DIR / "layers525-650CYLINDER6Updated.pt"),
    ("Cylinder7", DATA_DIR / "layers525-650CYLINDER7Updated.pt"),
    ("Cylinder10", DATA_DIR / "layers525-650CYLINDER10Updated.pt"),
    ("Cylinder11", DATA_DIR / "layers525-650CYLINDER11Updated.pt"),
    ("Cylinder12", DATA_DIR / "layers525-650CYLINDER12Updated.pt"),
    ("Cylinder13", DATA_DIR / "layers525-650CYLINDER13Updated.pt"),
    ("Cylinder14", DATA_DIR / "layers525-650CYLINDER14Updated.pt"),
    ("Cylinder15", DATA_DIR / "layers525-650CYLINDER15Updated.pt"),
    ("Cylinder18", DATA_DIR / "layers525-650CYLINDER18Updated.pt"),
    ("Cylinder19", DATA_DIR / "layers525-650CYLINDER19Updated.pt"),
    ("Cylinder26", DATA_DIR / "layers525-650CYLINDER26Updated.pt"),
    ("Cylinder27", DATA_DIR / "layers525-650CYLINDER27Updated.pt"),
    ("Cylinder28", DATA_DIR / "layers525-650CYLINDER28Updated.pt"),
    ("Cylinder44", DATA_DIR / "layers525-650CYLINDER44Updated.pt"),
    ("Cylinder47", DATA_DIR / "layers525-650CYLINDER47Updated.pt"),
]

RESUME_CHECKPOINT = DATA_DIR / "checkpoint_epoch400AdamWBatchNormDice.pt"
RESUME_EPOCH = 400

# ==========================================================
# INPUT CHANNEL SELECTION
# ==========================================================

# Available channels:
#
# 0 = IR0
# 1 = IR1
# 2 = IR2
# 3 = VIS0
# 4 = VIS1
# 5 = overMelting
# 6 = underMelting
# 7 = incompleteSpreading
# 8 = recoaterStreaking

INPUT_CHANNELS = [
    0,
    1
]
print("\nComputing global normalization...")

global_sum = torch.zeros(len(INPUT_CHANNELS))
global_sum_sq = torch.zeros(len(INPUT_CHANNELS))
global_count = 0

for _, file in training_files:
    raw = torch.load(file, map_location="cpu")

    for img in raw["X"]:
        img = img[INPUT_CHANNELS].float()
        global_sum += img.sum(dim=(1, 2))
        global_sum_sq += (img ** 2).sum(dim=(1, 2))
        global_count += img.shape[1] * img.shape[2]

    del raw


import gc



gc.collect()

loader_kwargs = {
    "batch_size": BATCH_SIZE,
    "num_workers": 0,
    "pin_memory": device.type == "cuda",
    "persistent_workers": False
}



model = uNet3D(2, 1, depth=WINDOW).to(device)

print("\nModel created.")


if RESUME_CHECKPOINT is not None:

    print(f"\nLoading checkpoint: {RESUME_CHECKPOINT}")

    checkpoint = torch.load(
        RESUME_CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"Checkpoint loaded successfully."
    )

    print(
        f"Resuming training from epoch {RESUME_EPOCH + 1}."
    )

else:

    print("\nStarting training from scratch.")

print(f"Input channels: ")

print(f"3D window depth: {WINDOW}")


total_pos = 0
total_pixels = 0

print("Computing pos_weight...")

for _, file in training_files:

    raw = torch.load(file, map_location="cpu")

    dataset = CylinderDataset3D(raw, window=WINDOW, augment=True, max_shift=MAX_SHIFT, input_channels=INPUT_CHANNELS)

    for label in dataset.Y:
        total_pos += label.sum().item()
        total_pixels += label.numel()

    del dataset
    del raw
    gc.collect()

pos_ratio = total_pos / total_pixels
pos_weight_value = (1.0 - pos_ratio) / pos_ratio
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

print(pos_weight.item())


print(f"\nPositive pixel ratio: "
      f"{pos_ratio:.8f}")

print(f"Computed pos_weight: "
      f"{pos_weight.item():.2f}")


crit = DiceBCELoss(pos_weight=pos_weight, dice_weight=1.0, bce_weight=1.0)


optim = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)

if device.type == "cuda":
    scaler = torch.amp.GradScaler("cuda")

else:

    scaler = None


startTime = time.time()

trainLoss = []




for epoch in range(RESUME_EPOCH, RESUME_EPOCH + EPOCHS):

    model.train()

    epochStart = time.time()

    epoch_loss = 0.0

    epoch_batches = 0

    print("\n" + "=" * 60)

    print(f"Starting Epoch "
          f"{epoch + 1}/{EPOCHS}")

    print("=" * 60)

    for cylinder_name, file in training_files:

        print(f"\nLoading {cylinder_name}")

        raw = torch.load(file, map_location="cpu")

        dataset = CylinderDataset3D(raw, window=WINDOW, augment=True, max_shift=MAX_SHIFT, input_channels=INPUT_CHANNELS)

        loader = DataLoader(dataset, shuffle=True, **loader_kwargs)

        del raw
        gc.collect()

        cylinder_loss = 0.0

        cylinder_batches = 0

        for b, (images, labels) in enumerate(loader):

            images = images.to(device,non_blocking=(device.type == "cuda"))

            labels = labels.to(device,non_blocking=(device.type == "cuda"))


            optim.zero_grad(set_to_none=True)

            if device.type == "cuda":

                with torch.amp.autocast(device_type="cuda",dtype=torch.float16):

                    labelPred = model(images)

                    loss = crit(

                        labelPred,

                        labels

                    )

                scaler.scale(loss).backward()

                scaler.step(optim)

                scaler.update()


            else:

                labelPred = model(images)

                loss = crit(

                    labelPred,

                    labels

                )

                loss.backward()

                optim.step()


            loss_value = (loss.item())

            epoch_loss += (loss_value)

            cylinder_loss += (loss_value)

            epoch_batches += 1

            cylinder_batches += 1


            if ((b + 1) % 10 == 0 or (b + 1) == len(loader)):
                print(

                    f"Epoch "
                    f"{epoch + 1}/{EPOCHS} | "

                    f"{cylinder_name} | "

                    f"Batch "
                    f"{b + 1}/{len(loader)} | "

                    f"Loss: "
                    f"{loss_value:.6f}"

                )

        if cylinder_batches > 0:

            average_cylinder_loss = (

                    cylinder_loss / cylinder_batches

            )

        else:

            average_cylinder_loss = 0.0

        print(

            f"{cylinder_name} "
            f"average loss: "
            f"{average_cylinder_loss:.6f}"

        )
        del loader
        del dataset

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------
    # EPOCH AVERAGE LOSS
    # ------------------------------------------

    average_epoch_loss = (

            epoch_loss / epoch_batches

    )

    trainLoss.append(

        average_epoch_loss

    )

    epochTime = (

            time.time() - epochStart

    )

    print(

        f"\nEpoch "
        f"{epoch + 1}/{EPOCHS} completed"

    )

    print(

        f"Average loss: "
        f"{average_epoch_loss:.6f}"

    )

    print(

        f"Epoch time: "
        f"{epochTime:.2f} seconds"

    )

    # ------------------------------------------
    # SAVE CHECKPOINT
    # ------------------------------------------

    if (

            (epoch + 1) % CHECKPOINT_INTERVAL == 0

    ):
        checkpoint_path = (

                DATA_DIR / (f"checkpoint_"
                            f"epoch{epoch + 1}.pt")

        )

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optim.state_dict(),
            "loss": average_epoch_loss, }, checkpoint_path)

        print(

            f"Saved checkpoint: "
            f"{checkpoint_path}"

        )

# ==========================================================
# TRAINING COMPLETE
# ==========================================================

totalTime = (

        time.time() - startTime

)

print("\n" + "=" * 60)

print("TRAINING COMPLETE")

print("=" * 60)

print(f"Total training time: "
      f"{totalTime:.2f} seconds")

print(f"Total training time: "
      f"{totalTime / 3600:.2f} hours")

final_model_path = (DATA_DIR / "final_model_e400BiggestAndDiverseDataset")

torch.save(

    model.state_dict(),

    final_model_path

)

print(

    f"Saved final model: "
    f"{final_model_path}"

)
