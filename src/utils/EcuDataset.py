from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.tensor_prep.create_windows import create_windows


class EcuDataset(Dataset):
    def __init__(self, data_path: Path, sequence_length: int = 64):
        flat_data = np.load(data_path).astype(np.float32)
        self.windows = create_windows(flat_data, sequence_length)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return torch.tensor(self.windows[index])
