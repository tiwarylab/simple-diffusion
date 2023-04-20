import torch
from torch.utils.data import Dataset
import numpy as np

class MinMaxTransform:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def standardize(self, data):
        std_data = (data - self.min)/(self.max - self.min)
        return std_data

    def unstandardize(self, std_data):
        data = std_data*(self.max - self.min) + self.min
        return data

class WhitenTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def standardize(self, data):
        std_data = (data - self.mean)/self.std
        return std_data

    def unstandardize(self, std_data):
        data = std_data*(self.std) + self.mean
        return data

TRANSFORM_DICT = {"minmax": MinMaxTransform,
                  "whiten": WhitenTransform}

class TorsionLoader(Dataset):
    def __init__(self, path: str,
                 n_channels: int = 1,
                 transform="whiten",
                 TRANSFORM_DICT=TRANSFORM_DICT):

        self.data = torch.from_numpy(np.load(path)).float()[:,1:]
        self.min = self.data.min(0)[0]
        self.max = self.data.max(0)[0]
        self.mean = self.data.mean(0)
        self.std = self.data.std(0)
        self.transform = TRANSFORM_DICT[transform](self.mean, self.std)
        self.data_std = self.transform.standardize(self.data)
        self.data_std = self.data_std.unsqueeze(1)

    def __getitem__(self, index):
        x = self.data_std[index]
        return x.float()

    def __len__(self):
        return self.data_std.shape[0]
