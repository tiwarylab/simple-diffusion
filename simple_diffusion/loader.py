import torch
from torch.utils.data import Dataset
import numpy as np

class TorsionLoader(Dataset):
    def __init__(self, path: str, n_channels: int = 1):
        self.data = torch.from_numpy(np.load(path)).float()[:,1:]
        self.data = self.data.unsqueeze(1)
        
    def __getitem__(self, index):
        x = self.data[index]
        return x.float()

    def __len__(self):
        return self.data.shape[0]
