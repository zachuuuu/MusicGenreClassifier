import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "dataset" / "raw" / "genres_original"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
SPECTROGRAMS_DIR = PROCESSED_DIR / "mel_spectrograms"
FEATURES_FILE = PROCESSED_DIR / "features.csv"


class MLPPreprocessor:

    def __init__(self, n_components=30):
        self.n_components = n_components
        self.n_mfcc = 20

    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path, duration=DURATION, sr=SAMPLE_RATE)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_var = mfcc.var(axis=1)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if np.ndim(tempo) > 0:
            tempo = tempo[0]

        features = np.concatenate([
            mfcc_mean,
            mfcc_var,
            [spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, tempo],
            chroma_mean
        ])
        return features

    def process_dataset(self):
        print("--- Rozpoczynam przetwarzanie danych dla MLP ---")
        data = []
        labels = []

        genres = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(RAW_DATA_DIR / d)]

        for genre in tqdm(genres, desc="Ekstrakcja cech MLP"):
            genre_dir = RAW_DATA_DIR / genre
            for file_name in os.listdir(genre_dir):
                if file_name.endswith('.wav'):
                    try:
                        features = self.extract_features(genre_dir / file_name)
                        data.append(features)
                        labels.append(genre)
                    except Exception as e:
                        print(f"Błąd: {file_name} - {e}")

        X = np.array(data)
        y = np.array(labels)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f"Liczba cech przed PCA: {X.shape[1]}")
        print(f"Nakładanie PCA (redukcja do {self.n_components} komponentów)...")
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(X_scaled)

        print(f"Zachowana wariancja: {np.sum(pca.explained_variance_ratio_):.2%}")

        df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(self.n_components)])
        df['label'] = y

        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df.to_csv(FEATURES_FILE, index=False)
        print(f"Dane dla MLP gotowe: {FEATURES_FILE}")


class CNNPreprocessor:

    def process_dataset(self):
        print("\n--- Rozpoczynam przetwarzanie danych dla CNN ---")

        first_shape = None

        if not os.path.exists(SPECTROGRAMS_DIR):
            os.makedirs(SPECTROGRAMS_DIR)

        genres = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(RAW_DATA_DIR / d)]

        target_len = SAMPLE_RATE * DURATION

        for genre in tqdm(genres, desc="Generowanie spektrogramów CNN"):
            genre_save_dir = SPECTROGRAMS_DIR / genre
            os.makedirs(genre_save_dir, exist_ok=True)
            genre_source_dir = RAW_DATA_DIR / genre

            for file_name in os.listdir(genre_source_dir):
                if file_name.endswith('.wav'):
                    try:
                        y, sr = librosa.load(genre_source_dir / file_name, sr=SAMPLE_RATE)

                        y = librosa.util.fix_length(y, size=target_len)

                        melspec = librosa.feature.melspectrogram(
                            y=y, sr=sr, n_mels=N_MELS
                        )
                        melspec_db = librosa.power_to_db(melspec, ref=np.max)

                        melspec_norm = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min())

                        if first_shape is None:
                            first_shape = melspec_norm.shape

                        np.save(genre_save_dir / f"{file_name[:-4]}.npy", melspec_norm)

                    except Exception as e:
                        print(f"Błąd: {file_name} - {e}")

        print(f"Shape mel-spectrogramów: {first_shape}")


if __name__ == "__main__":
    mlp_prep = MLPPreprocessor(n_components=30)
    mlp_prep.process_dataset()

    cnn_prep = CNNPreprocessor()
    cnn_prep.process_dataset()