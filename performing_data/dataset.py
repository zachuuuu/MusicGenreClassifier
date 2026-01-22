import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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
        n_cols = spec.shape[1]
        mask_width = np.random.randint(1, self.time_mask_param + 1)

        if mask_width < n_cols:
            mask_start = np.random.randint(0, n_cols - mask_width)
            spec[:, mask_start:mask_start + mask_width] = 0

        return spec

    def _freq_mask(self, spec):
        n_rows = spec.shape[0]
        mask_height = np.random.randint(1, self.freq_mask_param + 1)

        if mask_height < n_rows:
            mask_start = np.random.randint(0, n_rows - mask_height)
            spec[mask_start:mask_start + mask_height, :] = 0

        return spec


def prepare_mlp_data(csv_path, config):
    print("\n" + "=" * 60)
    print("PRZYGOTOWANIE DANYCH MLP")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values
    y_str = df['label'].values

    print(f"Wczytano {len(X)} próbek, {X.shape[1]} cech")

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    print(f"Klasy ({len(le.classes_)}): {list(le.classes_)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(config.VAL_SIZE + config.TEST_SIZE),
        random_state=config.SEED,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config.TEST_SIZE / (config.VAL_SIZE + config.TEST_SIZE),
        random_state=config.SEED,
        stratify=y_temp
    )

    print(f"Split: Train={len(X_train)} ({len(X_train) / len(X) * 100:.1f}%) | "
          f"Val={len(X_val)} ({len(X_val) / len(X) * 100:.1f}%) | "
          f"Test={len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), le


def prepare_cnn_data(spectrograms_dir, genres, config):
    print("\n" + "=" * 60)
    print("PRZYGOTOWANIE DANYCH CNN")
    print("=" * 60)

    file_paths = []
    labels = []

    for genre in genres:
        genre_dir = spectrograms_dir / genre
        if genre_dir.exists():
            for file_path in genre_dir.glob("*.npy"):
                file_paths.append(file_path)
                labels.append(genre)

    print(f"Znaleziono {len(file_paths)} spektrogramów")

    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"Klasy ({len(le.classes_)}): {list(le.classes_)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        file_paths, y,
        test_size=(config.VAL_SIZE + config.TEST_SIZE),
        random_state=config.SEED,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config.TEST_SIZE / (config.VAL_SIZE + config.TEST_SIZE),
        random_state=config.SEED,
        stratify=y_temp
    )

    print(f"Split: Train={len(X_train)} ({len(X_train) / len(file_paths) * 100:.1f}%) | "
          f"Val={len(X_val)} ({len(X_val) / len(file_paths) * 100:.1f}%) | "
          f"Test={len(X_test)} ({len(X_test) / len(file_paths) * 100:.1f}%)")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), le


def get_mlp_dataloaders(csv_path, config, batch_size=None):
    train_data, val_data, test_data, le = prepare_mlp_data(csv_path, config)

    train_dataset = MLPDataset(*train_data)
    val_dataset = MLPDataset(*val_data)
    test_dataset = MLPDataset(*test_data)

    final_batch_size = batch_size if batch_size is not None else config.MLP_CONFIG['batch_size']

    train_loader = DataLoader(
        train_dataset,
        batch_size=final_batch_size,
        shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader, le


def get_cnn_dataloaders(spectrograms_dir, genres, config, batch_size=None):
    train_data, val_data, test_data, le = prepare_cnn_data(spectrograms_dir, genres, config)

    train_dataset = CNNDataset(*train_data, augment=True, time_mask_param=20, freq_mask_param=10)
    val_dataset = CNNDataset(*val_data, augment=False)
    test_dataset = CNNDataset(*test_data, augment=False)

    print(f"Augmentacja włączona dla train set (SpecAugment)")

    final_batch_size = batch_size if batch_size is not None else config.CNN_CONFIG['batch_size']

    train_loader = DataLoader(
        train_dataset,
        batch_size=final_batch_size,
        shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader, le
