# 🎵 Music Genre Classifier

Celem projektu jest stworzenie i porównanie trzech architektur (**MLP**, **CNN** oraz **Random Forest**) do klasyfikacji gatunków muzycznych na podstawie zbioru danych **GTZAN**

##  Dataset
Wykorzystano zbiór **GTZAN Genre Collection**
* **Liczba utworów:** 1000 (po 100 na gatunek).
* **Format:** Pliki .wav, 30 sekund, 22050Hz.
* **Gatunki (10):** Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.

## Zastosowane Podejścia

### 1. **Random Forest** (Podejście oparte na cechach)

#### Preprocessing:
- **Ekstrakcja 57 cech audio:**
  - MFCC: mean + variance (40 cech)
  - Spectral Features: centroid, bandwidth, rolloff, ZCR (4 cechy)
  - Tempo: BPM (1 cecha)
  - Chroma: mean (12 cech)
- **Standaryzacja:** `StandardScaler`
- **Redukcja wymiarowości:** PCA do **30 komponentów** (zachowano **92.88%** wariancji)

#### Architektura:
- **N estimators**: 200
- **Max depth**: 30
- **Min samples split**: 2
- **Max features**: sqrt
- **Class weight**: balanced

### 2. Feature-based MLP (Podejście oparte na cechach)
#### Preprocessing:
- Identyczny jak Random Forest (30 komponentów PCA)

Optymalna architektura (dynamicznie dobierane warstwy ukryte) i hiperparametry zostały wybrane przez **Optuna**.

#### Architektura:
- Input (30) → Dense(1024) + BatchNorm + Dropout(0.536)
- Dense(512) + BatchNorm + Dropout(0.536)
- Dense(256) + BatchNorm + Dropout(0.536)
- Dense(10, softmax)

#### Regularyzacja:
- **Dropout:** 0.536
- **Weight Decay:** 9.28e-06
- **Early Stopping:** patience=10
- **Learning Rate Scheduler:** ReduceLROnPlateau

#### Optymalizator:
- **AdamW** (lr=0.00685)

### 3. Spectrogram CNN (Podejście oparte na obrazie)
#### Preprocessing:
- **Generowanie Mel-spektrogramów:**
  - 128 mel bins
  - 1292 time frames (30s × 22050 Hz)
  - Konwersja do skali dB
  - Normalizacja per-file: `(x - min) / (max - min)`
- **Augmentacja (SpecAugment):**
  - **Time Masking:** max 20 frames
  - **Frequency Masking:** max 10 bins
  - Losowe stosowanie (p=0.5)

Optymalna architektura (dynamicznie dobierane warstwy konwolucyjne) i hiperparametry zostały wybrane przez **Optuna**.

#### Architektura:
- Input (128×1292×1) → Conv2D(32) + BatchNorm + MaxPool2D + Dropout(0.496)
- Conv2D(64) + BatchNorm + MaxPool2D + Dropout(0.496)
- Conv2D(128) + BatchNorm + MaxPool2D + Dropout(0.496)
- Conv2D(256) + BatchNorm + MaxPool2D + Dropout(0.496)
- GlobalAveragePooling2D
- Dense(512) + BatchNorm + Dropout(0.496)
- Dense(10, softmax)

#### Regularyzacja:
- **Dropout:** 0.496
- **Weight Decay:** 2.19e-06
- **Early Stopping:** patience=10
- **Data Augmentation:** SpecAugment
- **Learning Rate Scheduler:** ReduceLROnPlateau

#### Optymalizator:
- **AdamW** (lr=0.000153)


## Wyniki modeli 

| Model | Test Accuracy | F1-Score | Precision | Recall | Czas Treningu |
|-------|---------------|----------|-----------|--------|---------------|
| **CNN** | **84.67%** | **84.08%** | **84.35%** | **84.67%** | 465.46s |
| **MLP** | **66.67%** | **65.77%** | **66.44%** | **66.67%** | 15.91s |
| **RF** | **61.33%** | **60.85%** | **61.53%** | **61.33%** | ~0.5s |

**CNN osiągnął najlepszy wynik** dzięki uczeniu przestrzennych wzorców z mel-spektrogramów. Warto zauważyć znaczną poprawę wydajności modelu MLP po migracji na TensorFlow.

## Porównanie każdej klasy (CNN - Najlepszy Model)

| Gatunek | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Blues | 87.5% | 93.3% | 90.3% |
| Classical | 93.3% | 93.3% | 93.3% |
| Country | 78.6% | 73.3% | 75.9% |
| Disco | 83.3% | 66.7% | 74.1% |
| Hip-hop | 81.2% | 86.7% | 83.9% |
| Jazz | 83.3% | 100.0% | 90.9% |
| Metal | 88.2% | 100.0% | 93.8% |
| Pop | 82.4% | 93.3% | 87.5% |
| Reggae | 92.9% | 86.7% | 89.7% |
| Rock | 72.7% | 53.3% | 61.5% |

**Obserwacje:**
- **Jazz i Metal** najłatwiejsze (100% recall)
- **Rock najtrudniejszy** (53% recall) - mylony z blues/country

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

**Dla modelu RF**:
```bash
python -m models.train --model rf
```

**Dla modelu MLP**:
```bash
python -m models.train --model mlp
```

**Dla modelu CNN**:
```bash
python -m models.train --model cnn
```
Wyniki (wykresy, macierze pomyłek, model .weights.h5 dla MLP/CNN lub .pkl dla RF) zostaną zapisane w folderze models/(mlp/cnn/rf)/reports/



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

### 5. API REST (FastAPI)
Nauczone modele można odpytywać przez interfejs webowy:
```bash
python -m api.app
```
lub
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Swagger UI** - http://localhost:8000/docs umożliwia wgranie pliku audio i wybór modelu do predykcji.

## Wykorzystane Technologie i Techniki

### Technologie
* **TensorFlow/Keras**: Budowa i trening sieci neuronowych.
* **Librosa**: Przetwarzanie sygnałów audio (MFCC, Mel-Spectrograms).
* **Scikit-learn**: PCA, skalowanie danych, metryki.
* **NumPy, Pandas**: Operacje numeryczne.
* **FastAPI**: Implementacja API REST.
* **Uvicorn**: ASGI Server.
* **Pydantic**: Walidacja danych.
* **Optuna**: Zaawansowana optymalizacja hiperparametrów (TPE Sampler, Pruning).
* **ClearML**: Śledzenie eksperymentów, logowanie metryk w chmurze.
* **Matplotlib, Seaborn**: Wykresy


### Techniki ML
**Preprocessing**
* **PCA**:  Redukcja wymiarowości (57 → 30)
* **StandardScaler**: Normalizacja cech
* **Mel-Spectrogram**: Reprezentacja time-frequency

**Augmentacja**
* **SpecAugment**: Time/Frequency Masking (CNN)

**Regularization**
* **Dropout**: Zapobieganie overfittingowi
* **Batch Normalization**: Stabilizacja treningu
* **Weight Decay (L2)**: Regularyzacja wag
* **Early Stopping**: Zatrzymanie przy braku poprawy

**Optymalizacja**
* **AdamW**: Optimizer z weight decay
* **ReduceLROnPlateau**: Dynamiczna redukcja LR
* **Class balancing**: Równe traktowanie klas (RF)

