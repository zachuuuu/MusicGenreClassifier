import argparse
import sys
import os
import time
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import config
import performing_data.dataset as ds
from models.mlp.mlp_model import MLP
from models.cnn.cnn_model import CNN
from models.rf.rf_model import RandomForest
import models.utils as utils


def setup_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_optimizer(cfg):
    opt_name = cfg['optimizer'].lower()
    lr = cfg['learning_rate']
    wd = cfg['weight_decay']

    if opt_name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == 'adamw':
        return tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=wd
        )
    elif opt_name == 'sgd':
        return tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=cfg.get('momentum', 0.9)
        )
    elif opt_name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        raise ValueError(f"Nieznany optymalizator: {opt_name}")


def main(model_type):
    if model_type == 'mlp':
        cfg = config.MLP_CONFIG
        reports_dir = config.MLP_REPORTS_DIR
        print(f"\nRozpoczynam trening modelu MLP: {config.PROJECT_NAME}")

        train_loader, val_loader, test_loader, le = ds.get_mlp_dataloaders(
            config.FEATURES_FILE, config
        )
        class_names = list(le.classes_)

        model = MLP(
            input_size=cfg['input_size'],
            hidden_sizes=cfg['hidden_sizes'],
            num_classes=config.NUM_CLASSES,
            dropout=cfg['dropout']
        )

    elif model_type == 'cnn':
        cfg = config.CNN_CONFIG
        reports_dir = config.CNN_REPORTS_DIR
        print(f"\nRozpoczynam trening modelu CNN: {config.PROJECT_NAME}")

        train_loader, val_loader, test_loader, le = ds.get_cnn_dataloaders(
            config.SPECTROGRAMS_DIR, config.GENRES, config
        )
        class_names = list(le.classes_)

        model = CNN(
            num_classes=config.NUM_CLASSES,
            conv_channels=cfg.get('conv_channels'),
            fc_size=cfg.get('fc_size', 256),
            dropout=cfg['dropout']
        )

    elif model_type == 'rf':
        cfg = config.RF_CONFIG
        reports_dir = config.RF_REPORTS_DIR
        print(f"\nRozpoczynam trening modelu RandomForest: {config.PROJECT_NAME}")

        train_data, val_data, test_data, le = ds.prepare_mlp_data(
            config.FEATURES_FILE, config
        )
        class_names = list(le.classes_)

        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        model = RandomForest(
            n_estimators=cfg['n_estimators'],
            max_depth=cfg['max_depth'],
            min_samples_split=cfg['min_samples_split'],
            min_samples_leaf=cfg['min_samples_leaf'],
            max_features=cfg['max_features'],
            random_state=cfg['random_state'],
            class_weight=cfg.get('class_weight')
        )

        reports_dir.mkdir(parents=True, exist_ok=True)
        print("Używane urządzenie: CPU (Random Forest)")

        print(f"\nStart treningu...")
        start_time = time.time()

        model.fit(X_train, y_train)

        total_time = time.time() - start_time
        print(f"Trening zakończony w {total_time:.2f}s")

        print("\nEwaluacja na zbiorze walidacyjnym...")
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average='weighted')
        val_precision = precision_score(y_val, val_preds, average='weighted')
        val_recall = recall_score(y_val, val_preds, average='weighted')

        print(f"Val Result -> Acc: {val_acc:.4f} | F1: {val_f1:.4f} | "
              f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")

        print("\nTestowanie modelu na zbiorze testowym...")
        test_preds = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='weighted')
        test_precision = precision_score(y_test, test_preds, average='weighted')
        test_recall = recall_score(y_test, test_preds, average='weighted')

        print(f"Test Result -> Acc: {test_acc:.4f} | F1: {test_f1:.4f} | "
              f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")

        utils.print_classification_report(y_test, test_preds, class_names)

        cm = confusion_matrix(y_test, test_preds)
        utils.plot_confusion_matrix(cm, class_names, save_path=reports_dir / "confusion_matrix.png")

        final_metrics = {
            'val_accuracy': float(val_acc),
            'val_f1': float(val_f1),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall),
            'test_accuracy': float(test_acc),
            'test_f1': float(test_f1),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'training_time': total_time,
            'n_estimators': cfg['n_estimators']
        }
        utils.save_metrics(final_metrics, reports_dir / "metrics.json")

        model.save_weights(reports_dir / "best_model.pkl")

        print(f"\nWszystkie wyniki zapisano w: {reports_dir}")
        return

    else:
        raise ValueError("Model musi być 'mlp', 'cnn' lub 'rf'")

    reports_dir.mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Używane urządzenie: GPU ({len(gpus)} device(s) found)")
        for gpu in gpus:
            print(f"  └─ {gpu.name}")
    else:
        print("Używane urządzenie: CPU (brak GPU)")

    optimizer = get_optimizer(cfg)
    print(f"Optimizer: {cfg['optimizer'].upper()} | LR: {cfg['learning_rate']} | Weight Decay: {cfg['weight_decay']}")

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=cfg['patience'],
            restore_best_weights=True,
            verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(reports_dir / "best_model.weights.h5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        ),
        utils.TrainingCallback(cfg['epochs'], val_loader)
    ]

    print(f"\nStart treningu ({cfg['epochs']} epok)...")
    start_time = time.time()

    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=cfg['epochs'],
        callbacks=callbacks,
        verbose=0
    )

    total_time = time.time() - start_time
    print(f"\nTrening zakończony w {total_time:.2f}s")

    history_dict = {
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_acc': history.history['accuracy'],
        'val_acc': history.history['val_accuracy']
    }

    print("\nGenerowanie raportów...")

    utils.plot_training_curves(history_dict, save_path=reports_dir / "training_curves.png")

    print("\nTestowanie najlepszego modelu na zbiorze testowym...")

    test_loss, test_acc = model.evaluate(test_loader, verbose=0)

    test_preds, test_labels = utils.get_predictions_and_labels(model, test_loader)

    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    test_precision = precision_score(test_labels, test_preds, average='weighted')
    test_recall = recall_score(test_labels, test_preds, average='weighted')

    print(f"Test Result -> Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | "
          f"F1: {test_f1:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")

    utils.print_classification_report(test_labels, test_preds, class_names)

    cm = confusion_matrix(test_labels, test_preds)
    utils.plot_confusion_matrix(cm, class_names, save_path=reports_dir / "confusion_matrix.png")

    final_metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'training_time': total_time,
        'epochs_trained': len(history_dict['train_loss']),
        'best_val_loss': float(min(history_dict['val_loss']))
    }
    utils.save_metrics(final_metrics, reports_dir / "metrics.json")

    print(f"\nWszystkie wyniki zapisano w: {reports_dir}")


if __name__ == "__main__":
    setup_seed(config.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['mlp', 'cnn', 'rf'])
    args = parser.parse_args()
    main(args.model)
