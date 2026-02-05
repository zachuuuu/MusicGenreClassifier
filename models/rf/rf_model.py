import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib


class RandomForest:

    def __init__(self, n_estimators=200, max_depth=30, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=42, class_weight=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
            verbose=0
        )
        self._is_trained = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self._is_trained = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save_weights(self, filepath):
        filepath = str(filepath).replace('.h5', '.pkl').replace('.keras', '.pkl').replace('.pth', '.pkl')
        joblib.dump(self.model, filepath)
        print(f"Model zapisany: {filepath}")

    def load_weights(self, filepath):
        filepath = str(filepath).replace('.h5', '.pkl').replace('.keras', '.pkl').replace('.pth', '.pkl')
        self.model = joblib.load(filepath)
        self._is_trained = True
        print(f"Model wczytany: {filepath}")
