import numpy as np
import librosa
import sys
import os
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import config
from performing_data.preprocess import MLPPreprocessor, SAMPLE_RATE, DURATION, N_MELS
from models.mlp.mlp_model import MLP
from models.cnn.cnn_model import CNN
from models.rf.rf_model import RandomForest


class ModelInference:

    def __init__(self):
        self.preprocessor = MLPPreprocessor(n_components=30)
        self.mlp_model = None
        self.cnn_model = None
        self.rf_model = None
        self.scaler = None
        self.pca = None
        self.models_loaded = False

    def load_models(self):
        print("Ładowanie modeli...")

        self.mlp_model = MLP(
            input_size=config.MLP_CONFIG['input_size'],
            hidden_sizes=config.MLP_CONFIG['hidden_sizes'],
            num_classes=config.NUM_CLASSES,
            dropout=config.MLP_CONFIG['dropout']
        )
        self.mlp_model(np.zeros((1, config.MLP_CONFIG['input_size'])))

        mlp_path = config.MLP_REPORTS_DIR / "best_model.weights.h5"
        self.mlp_model.load_weights(str(mlp_path))

        self.cnn_model = CNN(
            num_classes=config.NUM_CLASSES,
            conv_channels=config.CNN_CONFIG['conv_channels'],
            fc_size=config.CNN_CONFIG['fc_size'],
            dropout=config.CNN_CONFIG['dropout']
        )
        self.cnn_model(np.zeros((1, 128, 1292, 1)))

        cnn_path = config.CNN_REPORTS_DIR / "best_model.weights.h5"
        self.cnn_model.load_weights(str(cnn_path))

        self.rf_model = RandomForest()
        rf_path = config.RF_REPORTS_DIR / "best_model.pkl"
        self.rf_model.load_weights(str(rf_path))

        self.scaler = joblib.load(config.DATA_DIR / "scaler.pkl")
        self.pca = joblib.load(config.DATA_DIR / "pca.pkl")

        self.models_loaded = True
        print("Wszystkie modele zostały wczytane!")

    def extract_cnn_features_api(self, audio_path):
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        y = librosa.util.fix_length(y, size=SAMPLE_RATE * DURATION)

        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)

        melspec_norm = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min())

        return melspec_norm.reshape(1, 128, -1, 1).astype(np.float32)

    def predict_mlp(self, audio_path):
        raw_features = self.preprocessor.extract_features(audio_path)

        scaled_features = self.scaler.transform(raw_features.reshape(1, -1))
        pca_features = self.pca.transform(scaled_features)

        probs = self.mlp_model(pca_features, training=False).numpy()[0]
        return self._format_prediction(probs)

    def predict_cnn(self, audio_path):
        spectrogram = self.extract_cnn_features_api(audio_path)
        probs = self.cnn_model(spectrogram, training=False).numpy()[0]
        return self._format_prediction(probs)

    def predict_rf(self, audio_path):
        raw_features = self.preprocessor.extract_features(audio_path)

        scaled_features = self.scaler.transform(raw_features.reshape(1, -1))
        pca_features = self.pca.transform(scaled_features)

        probs = self.rf_model.predict_proba(pca_features)[0]

        return self._format_prediction(probs)

    def _format_prediction(self, probs):
        predicted_idx = int(np.argmax(probs))
        predicted_genre = config.GENRES[predicted_idx]
        confidence = float(probs[predicted_idx])

        probabilities = {
            genre: float(prob)
            for genre, prob in zip(config.GENRES, probs)
        }

        return {
            "predicted_genre": predicted_genre,
            "confidence": confidence,
            "probabilities": probabilities
        }


inference = ModelInference()
