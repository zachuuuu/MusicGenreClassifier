import torch
import numpy as np
from torch.utils.data import Dataset


class MLPDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CNNDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False,
                 time_mask_param=20, freq_mask_param=10):

        self.file_paths = file_paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        spectrogram = np.load(self.file_paths[idx])

        if self.augment:
            spectrogram = self._apply_specaugment(spectrogram)

        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)

        return spectrogram, self.labels[idx]

    def _apply_specaugment(self, spec):
        spec = spec.copy()

        if np.random.rand() < 0.5:
            spec = self._time_mask(spec)

        if np.random.rand() < 0.5:
            spec = self._freq_mask(spec)

        return spec

    def _time_mask(self, spec):
        n_cols = spec.shape[1]  # 1292
        mask_width = np.random.randint(1, self.time_mask_param + 1)

        if mask_width < n_cols:
            mask_start = np.random.randint(0, n_cols - mask_width)
            spec[:, mask_start:mask_start + mask_width] = 0

        return spec

    def _freq_mask(self, spec):
        n_rows = spec.shape[0]  # 128
        mask_height = np.random.randint(1, self.freq_mask_param + 1)

        if mask_height < n_rows:
            mask_start = np.random.randint(0, n_rows - mask_height)
            spec[mask_start:mask_start + mask_height, :] = 0

        return spec
