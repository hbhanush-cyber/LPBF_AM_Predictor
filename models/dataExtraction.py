import h5py
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

XCT_SCALE = 13752 / 2844

HDF5_FILE = r"C:\Users\hrida\Downloads\2024-05-01 M2 AMMTO Fatigue Blanks 05.hdf5"

IR0_PATH = "/slices/camera_data/nir/0"
IR1_PATH = "/slices/camera_data/nir/1"
IR2_PATH = "/slices/camera_data/nir/2"
VIS0_PATH = "/slices/camera_data/visible/0"
##VIS1_PATH = "/slices/camera_data/visible/1"
XCT_PATH = "/slices/registered_data/x-ray_ct_flaw"

class CylinderDataset(Dataset):

    def __init__(self, dataset):

        self.X = dataset["X"]
        self.Y = dataset["Y"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class Extractor:

    def __init__(
        self,
        xMin: int,
        xMax: int,
        yMin: int,
        yMax: int,
        layerMin: int,
        layerMax: int,
    ):
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.layerMin = layerMin
        self.layerMax = layerMax

    def extract(self, filename):

        X_data = []
        Y_data = []

        with h5py.File(HDF5_FILE, "r") as f:

            for i in range(self.layerMin, self.layerMax):


                ir0 = f[IR0_PATH][i]
                ir1 = f[IR1_PATH][i]
                ir2 = f[IR2_PATH][i]
                vis0 = f[VIS0_PATH][i]
                ##vis1 = f[VIS1_PATH][i]
                xct_full = f[XCT_PATH][i]


                ir0_c = ir0[self.yMin:self.yMax, self.xMin:self.xMax]
                ir1_c = ir1[self.yMin:self.yMax, self.xMin:self.xMax]
                ir2_c = ir2[self.yMin:self.yMax, self.xMin:self.xMax]
                vis0_c = vis0[self.yMin:self.yMax, self.xMin:self.xMax]
                ##vis1_c = vis1[self.yMin:self.yMax, self.xMin:self.xMax]


                xct_x_min = int(round(self.xMin * XCT_SCALE))
                xct_x_max = int(round(self.xMax * XCT_SCALE))
                xct_y_min = int(round(self.yMin * XCT_SCALE))
                xct_y_max = int(round(self.yMax * XCT_SCALE))

                xct_h, xct_w = xct_full.shape[:2]

                xct_x_min = max(0, xct_x_min)
                xct_y_min = max(0, xct_y_min)
                xct_x_max = min(xct_w, xct_x_max)
                xct_y_max = min(xct_h, xct_y_max)

                xct_cropped = xct_full[
                    xct_y_min:xct_y_max,
                    xct_x_min:xct_x_max
                ]

                new_width = ir0_c.shape[1]
                new_height = ir0_c.shape[0]

                xct_resized = cv2.resize(
                    xct_cropped,
                    (new_width, new_height),
                    interpolation=cv2.INTER_NEAREST
                )


                X = np.stack(
                    [
                        ir0_c,
                        ir1_c,
                        ir2_c,
                        vis0_c,
                        ##vis1_c,
                    ],
                    axis=-1,
                ).astype(np.float32)

                X = (
                    X - X.mean(axis=(0, 1), keepdims=True)
                ) / (
                    X.std(axis=(0, 1), keepdims=True) + 1e-8
                )

                X = np.transpose(X, (2, 0, 1))

                Y = xct_resized.astype(np.float32)
                Y = (Y > 0).astype(np.float32)
                Y = np.expand_dims(Y, axis=0)


                X_data.append(torch.from_numpy(X).float())
                Y_data.append(torch.from_numpy(Y).float())

        if filename is not None:
            save_path = Path.cwd() / f"{filename}.pt"

            torch.save(
                {
                    "X": X_data,
                    "Y": Y_data
                },
                save_path,
            )

        return {
            "X": X_data,
            "Y": Y_data,
        }


X_MIN = 2102
X_MAX = 2275
Y_MIN = 1910
Y_MAX = 2078

##trainExtractor = Extractor(xMin=2100, xMax=2282, yMin=786, yMax=968,layerMin=525,layerMax=650)
##dataset = CylinderDataset(trainExtractor.extract("layers525-650CYLINDER48"))
testExtractor = Extractor(xMin=X_MIN, xMax=X_MAX, yMin=Y_MIN, yMax=Y_MAX,layerMin=525,layerMax=650)
testdataset = CylinderDataset(testExtractor.extract("layers525-650CYLINDER24newpaddingtest"))