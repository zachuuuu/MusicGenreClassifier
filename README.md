# 🎵 Music Genre Classifier

Celem projektu jest stworzenie i porównanie dwóch architektur sieci neuronowych (MLP oraz CNN) do klasyfikacji gatunków muzycznych na podstawie zbioru danych **GTZAN**.


##  Dataset
Wykorzystano zbiór **GTZAN Genre Collection**
* **Liczba utworów:** 1000 (po 100 na gatunek).
* **Format:** Pliki .wav, 30 sekund, 22050Hz.
* **Gatunki (10):** Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.


## Zastosowane Podejścia

W projekcie zaimplementowano dwie różne ścieżki przetwarzania danych i architektury modeli:

### 1. Podejście oparte na cechach (Feature-based MLP)
* **Preprocessing:**
    * Ekstrakcja 57 cech audio: MFCC (40), Spectral Features (4), Tempo (1), Chroma (12).
    * Standaryzacja danych (`StandardScaler`).
    * Redukcja wymiarowości za pomocą **PCA** do 30 głównych komponentów (zachowano ~93% wariancji).
* **Architektura:**
    * Wielowarstwowy Perceptron (MLP) z dynamicznie dobieraną liczbą warstw ukrytych (konfigurowalna przez tuning).
    * Regularyzacja: **Dropout**, **BatchNorm**, **Weight Decay**, **Early Stopping**.
* **Wyniki:** Accuracy na zbiorze testowym: **~68%**.

### 2. Podejście oparte na obrazie (Spectrogram CNN)
* **Preprocessing:**
    * Generowanie **Mel-spektrogramów**.
    * Konwersja do skali decybelowej (dB).
    * Augmentacja danych w czasie rzeczywistym: **Time Masking** oraz **Frequency Masking** (SpecAugment).
* **Architektura:**
    * Konwolucyjna Sieć Neuronowa (CNN) z dynamiczną budową warstw konwolucyjnych (konfigurowalna przez tuning).
    * Wykorzystanie **Adaptive Average Pooling**, co pozwala na drastyczną redukcję parametrów i uniezależnienie od długości wejścia.
    * Regularyzacja: **Dropout**, **BatchNorm**, **Weight Decay**, **Early Stopping**.
* **Wyniki:** (Accuracy na zbiorze testowym **~85%**).

## Instalacja i Uruchomienie

### 1. Instalacja zależności
```bash
pip install -r requirements.txt
```

### 2. Przygotowanie danych (Preprocessing)
Skrypt pobiera surowe dane audio, generuje plik CSV z cechami oraz katalog ze spektrogramami.
```bash
python -m performing_data.preprocess
```

### 3. Trening modeli (Standardowy)
Uruchomienie treningu z parametrami zdefiniowanymi w config.py

**Dla modelu MLP**:
```bash
python -m models.train --model mlp
```

**Dla modelu CNN**:
```bash
python -m models.train --model cnn
```
Wyniki (wykresy, macierze pomyłek, model .weights.h5 + metadata) zostaną zapisane w folderze models/(mlp/cnn)/reports/


### 4. Hyperparameter Tuning (ClearML + Optuna)
Projekt obsługuje automatyczne poszukiwanie najlepszych parametrów (Learning Rate, Batch Size, Dropout, Architektura sieci). **Wymaga skonfigurowanego konta ClearML (clearml-init).**

**Dla modelu MLP**:
```bash
python -m models.tune --model mlp --trials ?
```

**Dla modelu CNN**:
```bash
python -m models.tune --model cnn --trials ?
```

## Wykorzystane Technologie i Techniki
* **TensorFlow/Keras**: Budowa i trening sieci neuronowych.
* **Librosa**: Przetwarzanie sygnałów audio (MFCC, Mel-Spectrograms).
* **Scikit-learn**: PCA, skalowanie danych, metryki.
* **Optuna**: Zaawansowana optymalizacja hiperparametrów (TPE Sampler, Pruning).
* **ClearML**: Śledzenie eksperymentów, logowanie metryk w chmurze.


* **Techniki ML**:
  * **PCA**
  * **Data Augmentation**
  * **Regularization** (Dropout, BatchNorm, Weight Decay, Early Stopping)
  * **Learning Rate Scheduler** (ReduceLROnPlateau)

