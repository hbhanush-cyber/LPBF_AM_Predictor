import torch
import numpy as np
import random

from CNNUNET3D import uNet3D
from CylinderDataset3D import CylinderDataset3D


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = (
    r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM"
    r"\models\final_model.pt"
)

TEST_DATA_PATH = (
    r"C:\Users\hrida\PycharmProjects\LBPF_ML_AM"
    r"\models\newTestNoV2real.pt"
)

WINDOW = 10

THRESHOLD = 0.5

NUM_SAMPLES = 15

SHIFTS = [
    0,
    5,
    10,
    20,
    40
]


# ============================================================
# SHIFT FUNCTION
# ============================================================

def shift_tensor(
    x,
    dx,
    dy,
    fill=0.0
):

    H = x.shape[-2]
    W = x.shape[-1]

    shifted = torch.full_like(
        x,
        fill
    )

    src_y0 = max(
        0,
        -dy
    )

    src_y1 = min(
        H,
        H - dy
    )

    src_x0 = max(
        0,
        -dx
    )

    src_x1 = min(
        W,
        W - dx
    )

    dst_y0 = max(
        0,
        dy
    )

    dst_y1 = min(
        H,
        H + dy
    )

    dst_x0 = max(
        0,
        dx
    )

    dst_x1 = min(
        W,
        W + dx
    )

    shifted[
        ...,
        dst_y0:dst_y1,
        dst_x0:dst_x1
    ] = x[
        ...,
        src_y0:src_y1,
        src_x0:src_x1
    ]

    return shifted


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth
):

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

    true_negative = np.logical_and(
        ~prediction,
        ~ground_truth
    ).sum()

    precision = (
        intersection /
        (predicted_pixels + 1e-8)
    )

    recall = (
        intersection /
        (actual_pixels + 1e-8)
    )

    iou = (
        intersection /
        (union + 1e-8)
    )

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-8)
    )

    pixel_accuracy = (
        prediction == ground_truth
    ).mean()

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
        "pixel_accuracy": pixel_accuracy
    }


# ============================================================
# LOAD DATA
# ============================================================

print(
    "Loading test dataset..."
)

raw_data = torch.load(
    TEST_DATA_PATH,
    map_location="cpu"
)

test_dataset = CylinderDataset3D(
    raw_data,
    window=WINDOW,
    augment=False
)

print(
    f"Number of layers: "
    f"{len(test_dataset)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "Loading model..."
)

model = uNet3D(
    4,
    1,
    depth=WINDOW
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print(
    "Model loaded successfully"
)


# ============================================================
# SELECT FIXED SAMPLES
# ============================================================

random.seed(
    42
)

indices = random.sample(
    range(
        len(test_dataset)
    ),
    min(
        NUM_SAMPLES,
        len(test_dataset)
    )
)


# ============================================================
# RESULTS
# ============================================================

results = {
    shift: []
    for shift in SHIFTS
}


# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    for sample_number, idx in enumerate(
        indices
    ):

        image, label = test_dataset[idx]

        print(
            f"Testing sample "
            f"{sample_number + 1}/"
            f"{len(indices)}"
        )

        for shift in SHIFTS:

            # =================================================
            # SHIFT ENTIRE 10-LAYER VOLUME
            # =================================================

            shifted_image = shift_tensor(
                image,
                dx=shift,
                dy=0,
                fill=0.0
            )

            # =================================================
            # SHIFT TARGET BY EXACT SAME AMOUNT
            # =================================================

            shifted_label = shift_tensor(
                label,
                dx=shift,
                dy=0,
                fill=0.0
            )

            # =================================================
            # ADD BATCH DIMENSION
            # =================================================

            image_batch = (
                shifted_image
                .unsqueeze(0)
                .to(DEVICE)
            )

            label_batch = (
                shifted_label
                .unsqueeze(0)
            )

            # =================================================
            # MODEL
            #
            # Output:
            #
            # (B,1,H,W)
            # =================================================

            prediction = torch.sigmoid(
                model(
                    image_batch
                )
            )

            binary_prediction = (
                prediction >
                THRESHOLD
            )

            # =================================================
            # REMOVE BATCH/CHANNEL
            # =================================================

            pred = (
                binary_prediction[
                    0,
                    0
                ]
                .cpu()
                .numpy()
            )

            gt = (
                label_batch[
                    0,
                    0
                ]
                .numpy()
            )

            # =================================================
            # METRICS
            # =================================================

            metrics = calculate_metrics(
                pred,
                gt
            )

            results[
                shift
            ].append(
                metrics
            )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)

print(
    "SHIFT ROBUSTNESS RESULTS"
)

print("=" * 70)

print(
    f"{'Shift':>10} | "
    f"{'Precision':>10} | "
    f"{'Recall':>10} | "
    f"{'IoU':>10} | "
    f"{'F1':>10}"
)

print(
    "-" * 70
)

for shift in SHIFTS:

    precision = np.mean([
        x["precision"]
        for x in results[shift]
    ])

    recall = np.mean([
        x["recall"]
        for x in results[shift]
    ])

    iou = np.mean([
        x["iou"]
        for x in results[shift]
    ])

    f1 = np.mean([
        x["f1"]
        for x in results[shift]
    ])

    print(
        f"{shift:>10} | "
        f"{precision:>10.4f} | "
        f"{recall:>10.4f} | "
        f"{iou:>10.4f} | "
        f"{f1:>10.4f}"
    )

print("=" * 70)