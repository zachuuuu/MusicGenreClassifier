from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "processed"
FEATURES_FILE = DATA_DIR / "features.csv"
SPECTROGRAMS_DIR = DATA_DIR / "mel_spectrograms"

MLP_REPORTS_DIR = BASE_DIR / "models" / "mlp" / "reports"
CNN_REPORTS_DIR = BASE_DIR / "models" / "cnn" / "reports"
RF_REPORTS_DIR = BASE_DIR / "models" / "rf" / "reports"

PROJECT_NAME = "MusicGenreClassifier_FDL"

NUM_CLASSES = 10
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
SEED = 42

TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

MLP_CONFIG = {
    'input_size': 30,
    'hidden_sizes': [1024, 512, 256],
    'dropout': 0.536,
    'optimizer': 'adamw',
    'learning_rate': 0.00685,
    'batch_size': 32,
    'epochs': 100,
    'weight_decay': 9.28e-06,
    'momentum': 0.9, # dla SGD
    'patience': 10
}

CNN_CONFIG = {
    'input_shape': (128, 1292, 1),
    'num_classes': 10,
    'conv_channels': [32, 64, 128, 256],
    'fc_size': 512,
    'dropout': 0.496,
    'optimizer': 'adamw',
    'learning_rate': 0.000153,
    'batch_size': 32,
    'epochs': 100,
    'weight_decay': 2.19e-06,
    'momentum': 0.9, # dla SGD
    'patience': 10
}

RF_CONFIG = {
    'n_estimators': 200,
    'max_depth': 30,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': SEED,
    'class_weight': 'balanced'
}