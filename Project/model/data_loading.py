import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import rioxarray as rioxr
from tqdm import tqdm

LABEL_MAP = {
    0: "Initial Fire",
    1: "DEM",
    2: "SB40",
    3: "FWI",
    4: "U-Wind",
    5: "V-Wind",
    6: "Gust",
    7: "Next Fire",
}


# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


class RasterDataset(Dataset):
    """
    This class extends the Pytorch Dataset object to our 8-channel fire raster images.
    """

    def __init__(self, raster_dir: str) -> None:
        """
        Initializes the data loader class. It takes in the directory where the raster images live,
        stores the directory as a class attribute, and checks that there were actually rasters
        recognized in that directory.
        """
        self.raster_dir = raster_dir

        self.raster_fnames = os.listdir(raster_dir)
        if not self.raster_fnames:
            raise RuntimeError(f'No data found in file path {raster_dir}')

        print(f'Creating dataset with {len(self.raster_fnames)} raster images')

    def __len__(self):
        """
        Returns the number of raster images loaded into the RasterDataset object
        """
        return len(self.raster_fnames)

    def __getitem__(self, index) -> T_co:
        pass


class RasterDataLoader:
    """
    This custom class loads in the accumulated raster data
    """

    def __init__(self, raster_dir: str, in_bands: int) -> None:
        """
        Initializes the data loader class. It takes in the directory where the raster images live,
        stores the directory as a class attribute, and checks that there were actually rasters
        recognized in that directory.
        """
        self.raster_dir = raster_dir
        self.in_bands = in_bands

        self.raster_fnames = os.listdir(raster_dir)
        if not self.raster_fnames:
            raise RuntimeError(f'No data found in file path {raster_dir}')

    def get_test_training_loaders(self) -> tuple:
        """
        This method initializes and returns train and test data loader objects created from the rasters
        in the directory provided to the self object when initialized.
        """
        X_path = os.path.join(self.raster_dir, 'X.npy')
        y_path = os.path.join(self.raster_dir, 'y.npy')
        if os.path.exists(X_path) and os.path.exists(y_path):
            X, y = np.load(X_path), np.load(y_path)
        else:
            X, y = self._get_data()
            np.save(X_path, X)
            np.save(y_path, y)

        return X, y

    def _get_data(self):
        """
        Loads all of the rasters in the raster directory to memory as numpy arrays
        """
        X, y = self._initialize_np_arrays()
        for i in tqdm(range(len(self.raster_fnames))):
            raster_name = self.raster_fnames[i]
            raster_path = os.path.join(self.raster_dir, raster_name)
            raster = rioxr.open_rasterio(raster_path)

            layers = raster.values[:-1]
            X[i, ...] = layers

            mask = raster.values[-1]
            y[i, ...] = mask
            pass
        return X, y

    def _initialize_np_arrays(self) -> tuple:
        """
        Initializes the np arrays based on the size of the raster images
        """
        for r_name in self.raster_fnames:
            rpath = os.path.join(self.raster_dir, r_name)
            raster = rioxr.open_rasterio(rpath)
            rows, cols = raster.values.shape[-2:]
            X = np.zeros([len(self.raster_fnames), self.in_bands, rows, cols], dtype=np.float32)
            y = np.zeros([len(self.raster_fnames), rows, cols], dtype=np.float32)
            return X, y
