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

MAX_SHIFT = 5

EPOCHS = 400

LEARNING_RATE = 1e-4

NUM_WORKERS = 2

CHECKPOINT_INTERVAL = 100

# ==========================================================
# DATA DIRECTORY
# ==========================================================

if Path("/content").exists():

    DATA_DIR = Path("/content/LBPF_ML_AM/models")

else:

    DATA_DIR = Path(r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models")


trainingDataFile1 = (DATA_DIR / "layers525-650CYLINDER1Updated.pt")
trainingDataFile8 = (DATA_DIR / "layers525-650CYLINDER8Updated.pt")
trainingDataFile9 = (DATA_DIR / "layers525-650CYLINDER9Updated.pt")
trainingDataFile16 = (DATA_DIR / "layers525-650CYLINDER16Updated.pt")
trainingDataFile17 = (DATA_DIR / "layers525-650CYLINDER17Updated.pt")
##trainingDataFile24 = (DATA_DIR / "layers525-650CYLINDER24Updated.pt")
trainingDataFile25 = (DATA_DIR / "layers525-650CYLINDER25Updated.pt")
trainingDataFile33 = (DATA_DIR / "layers525-650CYLINDER33Updated.pt")
trainingDataFile40 = (DATA_DIR / "layers525-650CYLINDER40Updated.pt")
trainingDataFile41 = (DATA_DIR / "layers525-650CYLINDER41Updated.pt")
trainingDataFile47 = (DATA_DIR / "layers525-650CYLINDER47Updated.pt")
trainingDataFile48 = (DATA_DIR / "layers525-650CYLINDER48Updated.pt")

testingDataFile = (DATA_DIR / "layers525-650CYLINDER24Updated.pt")


print("\nLoading datasets...")

rawData1 = torch.load(trainingDataFile1, map_location="cpu")
rawData8 = torch.load(trainingDataFile8, map_location="cpu")
rawData9 = torch.load(trainingDataFile9, map_location="cpu")
rawData16 = torch.load(trainingDataFile16, map_location="cpu")
rawData17 = torch.load(trainingDataFile17, map_location="cpu")
rawData25 = torch.load(trainingDataFile25, map_location="cpu")
rawData33 = torch.load(trainingDataFile33, map_location="cpu")
rawData40 = torch.load(trainingDataFile40, map_location="cpu")
rawData41 = torch.load(trainingDataFile41, map_location="cpu")
rawData47 = torch.load(trainingDataFile47, map_location="cpu")
rawData48 = torch.load(trainingDataFile48, map_location="cpu")

rawTestData = torch.load(testingDataFile, map_location="cpu")

dataset1 = CylinderDataset3D(rawData1, window=WINDOW, augment=True, max_shift=MAX_SHIFT)
dataset8 = CylinderDataset3D(rawData8, window=WINDOW, augment=True, max_shift=MAX_SHIFT)
dataset9 = CylinderDataset3D(rawData9, window=WINDOW, augment=True, max_shift=MAX_SHIFT)
dataset16 = CylinderDataset3D(rawData16, window=WINDOW, augment=True, max_shift=MAX_SHIFT)
dataset17 = CylinderDataset3D(rawData17, window=WINDOW, augment=True, max_shift=MAX_SHIFT)
dataset25 = CylinderDataset3D(rawData25, window=WINDOW, augment=True, max_shift=MAX_SHIFT)
dataset33 = CylinderDataset3D(rawData33, window=WINDOW, augment=True, max_shift=MAX_SHIFT)
dataset40 = CylinderDataset3D(rawData40, window=WINDOW, augment=True, max_shift=MAX_SHIFT)
dataset41 = CylinderDataset3D(rawData41, window=WINDOW, augment=True, max_shift=MAX_SHIFT)
dataset47 = CylinderDataset3D(rawData47, window=WINDOW, augment=True, max_shift=MAX_SHIFT)
dataset48 = CylinderDataset3D(rawData48, window=WINDOW, augment=True, max_shift=MAX_SHIFT)


testData = CylinderDataset3D(rawTestData, window=WINDOW, augment=False)

loader_kwargs = {"batch_size": BATCH_SIZE,"num_workers": NUM_WORKERS,"pin_memory": (device.type == "cuda"),"persistent_workers": (NUM_WORKERS > 0)}

train_loader1 = DataLoader(dataset1, shuffle=True, **loader_kwargs)
train_loader8 = DataLoader(dataset8, shuffle=True, **loader_kwargs)
train_loader9 = DataLoader(dataset9, shuffle=True, **loader_kwargs)
train_loader16 = DataLoader(dataset16, shuffle=True, **loader_kwargs)
train_loader17 = DataLoader(dataset17, shuffle=True, **loader_kwargs)
train_loader25 = DataLoader(dataset25, shuffle=True, **loader_kwargs)
train_loader33 = DataLoader(dataset33, shuffle=True, **loader_kwargs)
train_loader40 = DataLoader(dataset40, shuffle=True, **loader_kwargs)
train_loader41 = DataLoader(dataset41, shuffle=True, **loader_kwargs)
train_loader47 = DataLoader(dataset47, shuffle=True, **loader_kwargs)
train_loader48 = DataLoader(dataset48, shuffle=True, **loader_kwargs)



test_loader = DataLoader(testData,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=(device.type == "cuda"),persistent_workers=(NUM_WORKERS > 0))



model = uNet3D(9, 1, depth=WINDOW).to(device)

print("\nModel created.")

print(f"Input channels: ")

print(f"3D window depth: {WINDOW}")


training_datasets = [dataset1, dataset8, dataset9, dataset16, dataset25, dataset33, dataset40, dataset41, dataset47,]

total_pos = 0
total_pixels = 0

for ds in training_datasets:

    for i in range(len(ds)):
        total_pos += (ds.Y[i].sum().item())

        total_pixels += (ds.Y[i].numel())

pos_ratio = (total_pos / total_pixels)

pos_weight_value = ((1.0 - pos_ratio) / pos_ratio)

pos_weight = torch.tensor( [pos_weight_value], dtype=torch.float32, device=device)

print(f"\nPositive pixel ratio: "
      f"{pos_ratio:.8f}")

print(f"Computed pos_weight: "
      f"{pos_weight.item():.2f}")


crit = DiceBCELoss(pos_weight=pos_weight*0.5, dice_weight=0.75, bce_weight=1.0)


optim = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

if device.type == "cuda":
    scaler = torch.amp.GradScaler("cuda")

else:

    scaler = None


startTime = time.time()

trainLoss = []

train_loaders = [("Cylinder1", train_loader1),("Cylinder8", train_loader8),("Cylinder9",train_loader9), ("Cylinder16",train_loader16),("Cylinder17",train_loader17),("Cylinder25",train_loader25),("Cylinder33",train_loader33),("Cylinder40",train_loader40),("Cylinder41",train_loader41),("Cylinder47",train_loader47),("Cylinder48",train_loader48)]


for epoch in range(EPOCHS):

    model.train()

    epochStart = time.time()

    epoch_loss = 0.0

    epoch_batches = 0

    print("\n" + "=" * 60)

    print(f"Starting Epoch "
          f"{epoch + 1}/{EPOCHS}")

    print("=" * 60)


    for cylinder_name, loader in train_loaders:

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

# ==========================================================
# FINAL TEST EVALUATION
#
# No augmentation is applied to test data.
# ==========================================================

print("\n" + "=" * 60)

print("FINAL TEST EVALUATION")

print("=" * 60)

model.eval()

test_loss_total = 0.0

test_batches = 0

with torch.no_grad():
    for (testImages, testLabels) in test_loader:

        testImages = testImages.to(

            device,

            non_blocking=(device.type == "cuda")

        )

        testLabels = testLabels.to(

            device,

            non_blocking=(device.type == "cuda")

        )

        # --------------------------------------
        # FORWARD PASS
        # --------------------------------------

        if device.type == "cuda":

            with torch.amp.autocast(

                    device_type="cuda",

                    dtype=torch.float16

            ):

                labelVal = model(testImages)

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

            test_loss_total / test_batches

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

print("\n" + "=" * 60)

print("TRAINING SUMMARY")

print("=" * 60)

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

print("=" * 60)
