import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import time
from pathlib import Path

from CNNUNET3D import uNet3D
from CylinderDataset3D import CylinderDataset3D


# ============================================================
# SETTINGS
# ============================================================

MODEL_CHECKPOINT = r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\final_model_e300First3dCNNUnet10layerWindow4CylinderTraining.pt"

DATA_FILES = [
    r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\layers525-650CYLINDER8.pt",
    r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\layers525-650CYLINDER24.pt",
    r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\layers525-650CYLINDER40.pt",
    r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\layers525-650CYLINDER48.pt",
    r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM\models\layers525-650CYLINDER24newpaddingtest.pt"

]

WINDOW = 10
THRESHOLD = 0.5
BATCH_SIZE = 4
NUM_RANDOM_SAMPLES = 10

OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = uNet3D(4, 1)

model.load_state_dict(
    torch.load(
        MODEL_CHECKPOINT,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded successfully")


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(prediction, ground_truth):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    predicted_pixels = prediction.sum()
    actual_pixels = ground_truth.sum()

    precision = intersection / (
        predicted_pixels + 1e-8
    )

    recall = intersection / (
        actual_pixels + 1e-8
    )

    iou = intersection / (
        union + 1e-8
    )

    f1 = 2 * precision * recall / (
        precision + recall + 1e-8
    )

    pixel_accuracy = (
        prediction == ground_truth
    ).mean()

    return (
        precision,
        recall,
        iou,
        f1,
        pixel_accuracy
    )


# ============================================================
# EVALUATE ONE CYLINDER
# ============================================================

def evaluate_cylinder(data_file):

    print("\n" + "=" * 60)
    print(f"Evaluating:")
    print(data_file)
    print("=" * 60)

    t0 = time.time()

    raw_data = torch.load(
        data_file,
        map_location="cpu"
    )

    dataset = CylinderDataset3D(
        raw_data,
        window=WINDOW
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        f"Number of layers: {len(dataset)}"
    )

    all_precision = []
    all_recall = []
    all_iou = []
    all_f1 = []
    all_pixel_accuracy = []

    random_indices = random.sample(
        range(len(dataset)),
        min(
            NUM_RANDOM_SAMPLES,
            len(dataset)
        )
    )

    plot_data = {}

    total_correct = 0
    total_pixels = 0

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(loader):

            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            predictions = torch.sigmoid(
                predictions
            )

            # Only evaluate the LAST layer
            # in the 10-layer window

            predictions = predictions[
                :, :, -1, :, :
            ]

            labels = labels[
                :, :, -1, :, :
            ]

            binary_predictions = (
                predictions > THRESHOLD
            ).float()

            for j in range(
                images.shape[0]
            ):

                global_idx = (
                    batch_idx * BATCH_SIZE
                    + j
                )

                pred = (
                    binary_predictions[
                        j, 0
                    ]
                    .cpu()
                    .numpy()
                )

                raw_pred = (
                    predictions[
                        j, 0
                    ]
                    .cpu()
                    .numpy()
                )

                gt = (
                    labels[
                        j, 0
                    ]
                    .cpu()
                    .numpy()
                )

                precision, recall, iou, f1, pixel_accuracy = calculate_metrics(
                    pred,
                    gt
                )

                all_precision.append(
                    precision
                )

                all_recall.append(
                    recall
                )

                all_iou.append(
                    iou
                )

                all_f1.append(
                    f1
                )

                all_pixel_accuracy.append(
                    pixel_accuracy
                )

                total_correct += (
                    pred == gt
                ).sum()

                total_pixels += (
                    pred.size
                )

                if global_idx in random_indices:

                    image_cpu = (
                        images[
                            j
                        ]
                        .cpu()
                        .numpy()
                    )

                    plot_data[
                        global_idx
                    ] = (
                        image_cpu,
                        gt,
                        raw_pred,
                        pred
                    )

    # ========================================================
    # AVERAGE METRICS
    # ========================================================

    mean_precision = np.mean(
        all_precision
    )

    mean_recall = np.mean(
        all_recall
    )

    mean_iou = np.mean(
        all_iou
    )

    mean_f1 = np.mean(
        all_f1
    )

    mean_pixel_accuracy = np.mean(
        all_pixel_accuracy
    )

    print("\nResults")
    print("-" * 40)

    print(
        f"Precision:     {mean_precision:.4f}"
    )

    print(
        f"Recall:        {mean_recall:.4f}"
    )

    print(
        f"IoU:           {mean_iou:.4f}"
    )

    print(
        f"F1:            {mean_f1:.4f}"
    )

    print(
        f"Pixel Accuracy:{mean_pixel_accuracy:.4f}"
    )

    print(
        f"Evaluation Time: "
        f"{time.time() - t0:.2f} seconds"
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    cylinder_name = Path(
        data_file
    ).stem

    metrics_file = (
        OUTPUT_DIR
        / f"{cylinder_name}_metrics.txt"
    )

    with open(
        metrics_file,
        "w"
    ) as f:

        f.write(
            f"Cylinder: {cylinder_name}\n"
        )

        f.write(
            f"Layers: {len(dataset)}\n\n"
        )

        f.write(
            f"Precision: "
            f"{mean_precision:.6f}\n"
        )

        f.write(
            f"Recall: "
            f"{mean_recall:.6f}\n"
        )

        f.write(
            f"IoU: "
            f"{mean_iou:.6f}\n"
        )

        f.write(
            f"F1: "
            f"{mean_f1:.6f}\n"
        )

        f.write(
            f"Pixel Accuracy: "
            f"{mean_pixel_accuracy:.6f}\n"
        )


    # ========================================================
    # PLOT RANDOM SAMPLES
    # ========================================================

    for idx in random_indices:

        image, gt, raw_pred, binary_pred = (
            plot_data[idx]
        )

        fig, axes = plt.subplots(
            2,
            4,
            figsize=(16, 8)
        )

        axes[0, 0].imshow(
            image[0, -1],
            cmap="hot"
        )

        axes[0, 0].set_title(
            "IR Channel 0"
        )

        axes[0, 1].imshow(
            image[1, -1],
            cmap="hot"
        )

        axes[0, 1].set_title(
            "IR Channel 1"
        )

        axes[0, 2].imshow(
            image[2, -1],
            cmap="hot"
        )

        axes[0, 2].set_title(
            "IR Channel 2"
        )

        axes[0, 3].imshow(
            image[3, -1],
            cmap="magma"
        )

        axes[0, 3].set_title(
            "Visible Light"
        )

        axes[1, 0].imshow(
            gt,
            cmap="gray"
        )

        axes[1, 0].set_title(
            "Ground Truth"
        )

        axes[1, 1].imshow(
            raw_pred,
            cmap="viridis"
        )

        axes[1, 1].set_title(
            "Raw Prediction"
        )

        axes[1, 2].imshow(
            binary_pred,
            cmap="gray"
        )

        axes[1, 2].set_title(
            f"Binary Prediction "
            f"(t={THRESHOLD})"
        )

        axes[1, 3].imshow(
            gt,
            cmap="gray"
        )

        axes[1, 3].imshow(
            binary_pred,
            cmap="Reds",
            alpha=0.5
        )

        axes[1, 3].set_title(
            "Prediction vs Ground Truth"
        )

        for ax in axes.flat:
            ax.axis("off")

        plt.suptitle(
            f"{cylinder_name} - Layer {idx}"
        )

        plt.tight_layout()

        output_file = (
            OUTPUT_DIR
            / f"{cylinder_name}_layer_{idx}.png"
        )

        plt.savefig(
            output_file,
            dpi=150
        )

        plt.close()


    return {
        "cylinder": cylinder_name,
        "precision": mean_precision,
        "recall": mean_recall,
        "iou": mean_iou,
        "f1": mean_f1,
        "pixel_accuracy": mean_pixel_accuracy
    }


# ============================================================
# EVALUATE ALL CYLINDERS
# ============================================================

all_results = []

for data_file in DATA_FILES:

    result = evaluate_cylinder(
        data_file
    )

    all_results.append(
        result
    )


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL EVALUATION SUMMARY")
print("=" * 70)

for result in all_results:

    print(
        f"\n{result['cylinder']}"
    )

    print(
        f"Precision:      "
        f"{result['precision']:.4f}"
    )

    print(
        f"Recall:         "
        f"{result['recall']:.4f}"
    )

    print(
        f"IoU:            "
        f"{result['iou']:.4f}"
    )

    print(
        f"F1:             "
        f"{result['f1']:.4f}"
    )

    print(
        f"Pixel Accuracy: "
        f"{result['pixel_accuracy']:.4f}"
    )

print("\nEvaluation complete.")
print(
    f"All results saved to: "
    f"{OUTPUT_DIR.absolute()}"
)