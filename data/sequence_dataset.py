import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):

    def __init__(self, X, Y, store_ids, window_size=7):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.window_size = window_size
        self.index = self._build_index(store_ids)

    def _build_index(self, store_ids):
        index = []
        for store_id in np.unique(store_ids):
            store_positions = np.where(store_ids == store_id)[0]
            if len(store_positions) <= self.window_size:
                continue
            for i in range(len(store_positions) - self.window_size):
                window_start = store_positions[i]
                target_pos   = store_positions[i + self.window_size]
                index.append((window_start, target_pos))
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        window_start, target_pos = self.index[idx]
        X_window = self.X[window_start : window_start + self.window_size]
        Y_target = self.Y[target_pos]
        return X_window, Y_target