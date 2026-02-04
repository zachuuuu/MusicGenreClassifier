import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def apply_specaugment(spec, time_mask_param=20, freq_mask_param=10):
    spec = spec.copy()

    if np.random.rand() < 0.5:
        n_cols = spec.shape[1]
        mask_width = np.random.randint(1, time_mask_param + 1)
        if mask_width < n_cols:
            mask_start = np.random.randint(0, n_cols - mask_width)
            spec[:, mask_start:mask_start + mask_width, :] = 0

    if np.random.rand() < 0.5:
        n_rows = spec.shape[0]
        mask_height = np.random.randint(1, freq_mask_param + 1)
        if mask_height < n_rows:
            mask_start = np.random.randint(0, n_rows - mask_height)
            spec[mask_start:mask_start + mask_height, :, :] = 0

    return spec


def load_spectrogram(file_path, label, augment=False):

    def _load_npy(path):
        spec = np.load(path.numpy().decode('utf-8'))
        spec = np.expand_dims(spec, axis=-1).astype(np.float32)

        if augment:
            spec = apply_specaugment(spec, time_mask_param=20, freq_mask_param=10)

        return spec

    spectrogram = tf.py_function(_load_npy, [file_path], tf.float32)
    spectrogram.set_shape([128, 1292, 1])

    return spectrogram, label


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
                file_paths.append(str(file_path))
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

    final_batch_size = batch_size if batch_size is not None else config.MLP_CONFIG['batch_size']

    X_train = train_data[0].astype(np.float32)
    y_train = train_data[1].astype(np.int32)
    X_val = val_data[0].astype(np.float32)
    y_val = val_data[1].astype(np.int32)
    X_test = test_data[0].astype(np.float32)
    y_test = test_data[1].astype(np.int32)

    train_loader = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_loader = train_loader.shuffle(len(X_train)).batch(final_batch_size).prefetch(tf.data.AUTOTUNE)

    val_loader = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_loader = val_loader.batch(64).prefetch(tf.data.AUTOTUNE)

    test_loader = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_loader = test_loader.batch(64).prefetch(tf.data.AUTOTUNE)

    return train_loader, val_loader, test_loader, le


def get_cnn_dataloaders(spectrograms_dir, genres, config, batch_size=None):
    train_data, val_data, test_data, le = prepare_cnn_data(spectrograms_dir, genres, config)

    final_batch_size = batch_size if batch_size is not None else config.CNN_CONFIG['batch_size']

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)

    train_loader = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_loader = train_loader.shuffle(len(X_train))
    train_loader = train_loader.map(
        lambda x, y: load_spectrogram(x, y, augment=True),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_loader = train_loader.batch(final_batch_size).prefetch(tf.data.AUTOTUNE)

    val_loader = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_loader = val_loader.map(
        lambda x, y: load_spectrogram(x, y, augment=False),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_loader = val_loader.batch(16).prefetch(tf.data.AUTOTUNE)

    test_loader = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_loader = test_loader.map(
        lambda x, y: load_spectrogram(x, y, augment=False),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_loader = test_loader.batch(16).prefetch(tf.data.AUTOTUNE)

    print(f"Augmentacja włączona dla train set (SpecAugment)")

    return train_loader, val_loader, test_loader, le
