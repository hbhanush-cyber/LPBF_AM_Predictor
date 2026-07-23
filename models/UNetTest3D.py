import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CNNUNET3D import uNet3D
from CylinderDataset3D import CylinderDataset3D


# ==========================================================
# LOSS FUNCTION
# ==========================================================

class DiceBCELoss(nn.Module):

    def __init__(
        self,
        pos_weight=None,
        dice_weight=1.0,
        bce_weight=1.0,
        smooth=1.0
    ):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth


    def forward(self, logits, targets):

        # ------------------------------------------
        # BCE LOSS
        # ------------------------------------------

        bce_loss = self.bce(
            logits,
            targets
        )


        # ------------------------------------------
        # DICE LOSS
        # ------------------------------------------

        probs = torch.sigmoid(
            logits
        )

        probs_flat = probs.view(
            probs.size(0),
            -1
        )

        targets_flat = targets.view(
            targets.size(0),
            -1
        )

        intersection = (
            probs_flat * targets_flat
        ).sum(dim=1)

        dice_score = (
            2.0 * intersection
            + self.smooth
        ) / (
            probs_flat.sum(dim=1)
            + targets_flat.sum(dim=1)
            + self.smooth
        )

        dice_loss = (
            1.0 - dice_score.mean()
        )


        # ------------------------------------------
        # COMBINED LOSS
        # ------------------------------------------

        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss


# ==========================================================
# DEVICE
# ==========================================================

if torch.cuda.is_available():

    device = torch.device(
        "cuda"
    )

else:

    device = torch.device(
        "cpu"
    )


print(
    f"Using device: {device}"
)


if device.type == "cuda":

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# ==========================================================
# SETTINGS
# ==========================================================

WINDOW = 10

BATCH_SIZE = 32

MAX_SHIFT = 5

EPOCHS = 100

LEARNING_RATE = 1e-4

NUM_WORKERS = 2

CHECKPOINT_INTERVAL = 10


# ==========================================================
# DATA DIRECTORY
# ==========================================================

if Path("/content").exists():

    DATA_DIR = Path(
        "/content/LBPF_ML_AM/models"
    )

else:

    DATA_DIR = Path(
        r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models"
    )


# ==========================================================
# DATA FILES
# ==========================================================

trainingDataFile = (
    DATA_DIR
    / "layers525-650CYLINDER24.pt"
)

trainingDataFile1 = (
    DATA_DIR
    / "layers525-650CYLINDER48.pt"
)

trainingDataFile2 = (
    DATA_DIR
    / "layers525-650CYLINDER40.pt"
)

trainingDataFile3 = (
    DATA_DIR
    / "layers525-650CYLINDER8.pt"
)

testingDataFile = (
    DATA_DIR
    / "newTestNoV2real.pt"
)


# ==========================================================
# LOAD RAW DATA
# ==========================================================

print(
    "\nLoading datasets..."
)


rawData24 = torch.load(
    trainingDataFile,
    map_location="cpu"
)

rawData48 = torch.load(
    trainingDataFile1,
    map_location="cpu"
)

rawData40 = torch.load(
    trainingDataFile2,
    map_location="cpu"
)

rawData8 = torch.load(
    trainingDataFile3,
    map_location="cpu"
)

rawTestData = torch.load(
    testingDataFile,
    map_location="cpu"
)


# ==========================================================
# CREATE TRAINING DATASETS
#
# Each cylinder remains a separate dataset because
# the original cylinders have different spatial dimensions.
#
# Shift augmentation is applied only to training data.
# ==========================================================

dataset24 = CylinderDataset3D(
    rawData24,
    window=WINDOW,
    augment=True,
    max_shift=MAX_SHIFT
)

dataset48 = CylinderDataset3D(
    rawData48,
    window=WINDOW,
    augment=True,
    max_shift=MAX_SHIFT
)

dataset40 = CylinderDataset3D(
    rawData40,
    window=WINDOW,
    augment=True,
    max_shift=MAX_SHIFT
)

dataset8 = CylinderDataset3D(
    rawData8,
    window=WINDOW,
    augment=True,
    max_shift=MAX_SHIFT
)


# ==========================================================
# CREATE TEST DATASET
#
# IMPORTANT:
# No augmentation is used for testing.
# ==========================================================

testData = CylinderDataset3D(
    rawTestData,
    window=WINDOW,
    augment=False
)


# ==========================================================
# DATA LOADERS
# ==========================================================

loader_kwargs = {

    "batch_size": BATCH_SIZE,

    "num_workers": NUM_WORKERS,

    "pin_memory": (
        device.type == "cuda"
    ),

    "persistent_workers": (
        NUM_WORKERS > 0
    )
}


train_loader24 = DataLoader(
    dataset24,
    shuffle=True,
    **loader_kwargs
)


train_loader48 = DataLoader(
    dataset48,
    shuffle=True,
    **loader_kwargs
)


train_loader40 = DataLoader(
    dataset40,
    shuffle=True,
    **loader_kwargs
)


train_loader8 = DataLoader(
    dataset8,
    shuffle=True,
    **loader_kwargs
)


test_loader = DataLoader(

    testData,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=(
        device.type == "cuda"
    ),

    persistent_workers=(
        NUM_WORKERS > 0
    )
)


# ==========================================================
# DATASET INFORMATION
# ==========================================================

print(
    "\nDataset sizes:"
)

print(
    f"Cylinder 24: "
    f"{len(dataset24)} layers"
)

print(
    f"Cylinder 48: "
    f"{len(dataset48)} layers"
)

print(
    f"Cylinder 40: "
    f"{len(dataset40)} layers"
)

print(
    f"Cylinder 8: "
    f"{len(dataset8)} layers"
)

print(
    f"Test: "
    f"{len(testData)} layers"
)


print(
    "\nNumber of batches:"
)

print(
    f"Cylinder 24: "
    f"{len(train_loader24)}"
)

print(
    f"Cylinder 48: "
    f"{len(train_loader48)}"
)

print(
    f"Cylinder 40: "
    f"{len(train_loader40)}"
)

print(
    f"Cylinder 8: "
    f"{len(train_loader8)}"
)

print(
    f"Test: "
    f"{len(test_loader)}"
)


# ==========================================================
# CREATE MODEL
#
# Input:
#   4 channels
#   10 previous/current layers
#
# Shape:
#   (B, 4, 10, H, W)
#
# Output:
#   (B, 1, H, W)
# ==========================================================

model = uNet3D(
    4,
    1,
    depth=WINDOW
).to(device)


print(
    "\nModel created."
)

print(
    f"Input channels: 4"
)

print(
    f"3D window depth: {WINDOW}"
)


# ==========================================================
# CALCULATE CLASS IMBALANCE
#
# The shift augmentation does not change the number
# of positive pixels, so calculating pos_weight from
# the original labels is valid.
# ==========================================================

training_datasets = [

    dataset24,

    dataset48,

    dataset40,

    dataset8

]


total_pos = 0

total_pixels = 0


for ds in training_datasets:

    for i in range(
        len(ds)
    ):

        total_pos += (
            ds.Y[i]
            .sum()
            .item()
        )

        total_pixels += (
            ds.Y[i]
            .numel()
        )


pos_ratio = (
    total_pos
    / total_pixels
)


pos_weight_value = (
    (1.0 - pos_ratio)
    / pos_ratio
)


pos_weight = torch.tensor(

    [pos_weight_value],

    dtype=torch.float32,

    device=device

)


print(
    f"\nPositive pixel ratio: "
    f"{pos_ratio:.8f}"
)

print(
    f"Computed pos_weight: "
    f"{pos_weight.item():.2f}"
)


# ==========================================================
# LOSS
# ==========================================================

crit = DiceBCELoss(

    pos_weight=pos_weight,

    dice_weight=1.0,

    bce_weight=1.0

)


# ==========================================================
# OPTIMIZER
# ==========================================================

optim = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)


# ==========================================================
# MIXED PRECISION
#
# The T4 GPU benefits from mixed precision.
# ==========================================================

if device.type == "cuda":

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ==========================================================
# TRAINING SETUP
# ==========================================================

startTime = time.time()


trainLoss = []


# Keep the four loaders separate.
#
# This is necessary because the four cylinders have
# different spatial dimensions.
#
# Each epoch still trains on every cylinder.

train_loaders = [

    (
        "Cylinder24",
        train_loader24
    ),

    (
        "Cylinder48",
        train_loader48
    ),

    (
        "Cylinder40",
        train_loader40
    ),

    (
        "Cylinder8",
        train_loader8
    )

]


# ==========================================================
# TRAINING LOOP
# ==========================================================

for epoch in range(
    EPOCHS
):

    model.train()


    epochStart = time.time()


    epoch_loss = 0.0

    epoch_batches = 0


    print(
        "\n"
        + "=" * 60
    )

    print(
        f"Starting Epoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    print(
        "=" * 60
    )


    # ------------------------------------------
    # TRAIN ON EACH CYLINDER
    # ------------------------------------------

    for cylinder_name, loader in train_loaders:

        cylinder_loss = 0.0

        cylinder_batches = 0


        for b, (
            images,
            labels
        ) in enumerate(loader):


            # --------------------------------------
            # MOVE DATA TO GPU
            # --------------------------------------

            images = images.to(

                device,

                non_blocking=(
                    device.type == "cuda"
                )

            )


            labels = labels.to(

                device,

                non_blocking=(
                    device.type == "cuda"
                )

            )


            # --------------------------------------
            # RESET GRADIENTS
            # --------------------------------------

            optim.zero_grad(
                set_to_none=True
            )


            # --------------------------------------
            # FORWARD PASS
            # --------------------------------------

            if device.type == "cuda":

                with torch.amp.autocast(

                    device_type="cuda",

                    dtype=torch.float16

                ):

                    labelPred = model(
                        images
                    )

                    loss = crit(

                        labelPred,

                        labels

                    )


                # ----------------------------------
                # BACKWARD PASS
                # ----------------------------------

                scaler.scale(
                    loss
                ).backward()


                scaler.step(
                    optim
                )


                scaler.update()


            else:

                labelPred = model(
                    images
                )

                loss = crit(

                    labelPred,

                    labels

                )


                loss.backward()


                optim.step()


            # --------------------------------------
            # RECORD LOSS
            # --------------------------------------

            loss_value = (
                loss.item()
            )


            epoch_loss += (
                loss_value
            )


            cylinder_loss += (
                loss_value
            )


            epoch_batches += 1

            cylinder_batches += 1


            # --------------------------------------
            # PROGRESS
            # --------------------------------------

            if (
                (b + 1) % 10 == 0
                or
                (b + 1) == len(loader)
            ):

                print(

                    f"Epoch "
                    f"{epoch + 1}/{EPOCHS} | "

                    f"{cylinder_name} | "

                    f"Batch "
                    f"{b + 1}/{len(loader)} | "

                    f"Loss: "
                    f"{loss_value:.6f}"

                )


        # ------------------------------------------
        # CYLINDER AVERAGE LOSS
        # ------------------------------------------

        if cylinder_batches > 0:

            average_cylinder_loss = (

                cylinder_loss
                / cylinder_batches

            )

        else:

            average_cylinder_loss = 0.0


        print(

            f"{cylinder_name} "
            f"average loss: "
            f"{average_cylinder_loss:.6f}"

        )


    # ------------------------------------------
    # EPOCH AVERAGE LOSS
    # ------------------------------------------

    average_epoch_loss = (

        epoch_loss
        / epoch_batches

    )


    trainLoss.append(

        average_epoch_loss

    )


    epochTime = (

        time.time()
        - epochStart

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

        (epoch + 1)
        % CHECKPOINT_INTERVAL
        == 0

    ):

        checkpoint_path = (

            DATA_DIR
            / (
                f"checkpoint_"
                f"epoch{epoch + 1}.pt"
            )

        )


        torch.save(

            model.state_dict(),

            checkpoint_path

        )


        print(

            f"Saved checkpoint: "
            f"{checkpoint_path}"

        )


# ==========================================================
# TRAINING COMPLETE
# ==========================================================

totalTime = (

    time.time()
    - startTime

)


print(
    "\n"
    + "=" * 60
)

print(
    "TRAINING COMPLETE"
)

print(
    "=" * 60
)

print(
    f"Total training time: "
    f"{totalTime:.2f} seconds"
)

print(
    f"Total training time: "
    f"{totalTime / 3600:.2f} hours"
)


# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

final_model_path = (

    DATA_DIR
    / "final_model_e100_3dCNNUnet10layerWindow.pt"

)


torch.save(

    model.state_dict(),

    final_model_path

)


print(

    f"Saved final model: "
    f"{final_model_path}"

)


# ==========================================================
# FINAL TEST EVALUATION
#
# No augmentation is applied to test data.
# ==========================================================

print(
    "\n"
    + "=" * 60
)

print(
    "FINAL TEST EVALUATION"
)

print(
    "=" * 60
)


model.eval()


test_loss_total = 0.0

test_batches = 0


with torch.no_grad():

    for (
        testImages,
        testLabels
    ) in test_loader:


        testImages = testImages.to(

            device,

            non_blocking=(
                device.type == "cuda"
            )

        )


        testLabels = testLabels.to(

            device,

            non_blocking=(
                device.type == "cuda"
            )

        )


        # --------------------------------------
        # FORWARD PASS
        # --------------------------------------

        if device.type == "cuda":

            with torch.amp.autocast(

                device_type="cuda",

                dtype=torch.float16

            ):

                labelVal = model(
                    testImages
                )

                lossTest = crit(

                    labelVal,

                    testLabels

                )

        else:

            labelVal = model(

                testImages

            )

            lossTest = crit(

                labelVal,

                testLabels

            )


        test_loss_total += (

            lossTest.item()

        )


        test_batches += 1


# ==========================================================
# TEST LOSS
# ==========================================================

if test_batches > 0:

    average_test_loss = (

        test_loss_total
        / test_batches

    )

else:

    average_test_loss = 0.0


print(

    f"Test loss: "
    f"{average_test_loss:.6f}"

)


# ==========================================================
# TRAINING SUMMARY
# ==========================================================

print(
    "\n"
    + "=" * 60
)

print(
    "TRAINING SUMMARY"
)

print(
    "=" * 60
)

print(

    f"Epochs trained: "
    f"{EPOCHS}"

)

print(

    f"Window size: "
    f"{WINDOW}"

)

print(

    f"Maximum training shift: "
    f"{MAX_SHIFT} pixels"

)

print(

    f"Batch size: "
    f"{BATCH_SIZE}"

)

print(

    f"Learning rate: "
    f"{LEARNING_RATE}"

)

print(

    f"Final training loss: "
    f"{trainLoss[-1]:.6f}"

)

print(

    f"Final test loss: "
    f"{average_test_loss:.6f}"

)

print(

    f"Final model: "
    f"{final_model_path}"

)

print(
    "=" * 60
)