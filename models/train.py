import argparse
import sys
import os
import time
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import config
import performing_data.dataset as ds
from models.mlp.mlp_model import MLP
from models.cnn.cnn_model import CNN
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
            dropout=cfg['dropout'],
            weight_decay=cfg['weight_decay']
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
            dropout=cfg['dropout'],
            weight_decay=cfg['weight_decay']
        )
    else:
        raise ValueError("Model musi być 'mlp' lub 'cnn'")

    reports_dir.mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Używane urządzenie: GPU ({len(gpus)} device(s) found)")
        for gpu in gpus:
            print(f"  └─ {gpu.name}")
    else:
        print("Używane urządzenie: CPU (brak GPU)")

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = get_optimizer(cfg)
    print(f"Optimizer: {cfg['optimizer'].upper()} | LR: {cfg['learning_rate']} | Weight Decay: {cfg['weight_decay']}")

    best_val_loss = float('inf')
    patience_counter = 0

    early_stopping = utils.EarlyStopping(patience=cfg['patience'], mode='min')

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\nStart treningu ({cfg['epochs']} epok)...")
    start_time = time.time()

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss, train_acc = utils.train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc, val_f1, val_precision, val_recall, _, _ = utils.evaluate_model(
            model, val_loader, criterion
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                old_lr = optimizer.learning_rate.numpy()
                new_lr = max(old_lr * 0.5, 1e-6)
                optimizer.learning_rate.assign(new_lr)
                print(f"ReduceLROnPlateau: LR zmieniony z {old_lr:.6f} na {new_lr:.6f}")
                patience_counter = 0

        current_lr = optimizer.learning_rate.numpy()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{cfg['epochs']} | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if early_stopping(val_loss):
            utils.save_model(
                model, epoch,
                {'val_loss': val_loss, 'val_acc': val_acc},
                reports_dir / "best_model.pth"
            )

        if early_stopping.early_stop:
            print(f"\nEarly Stopping zadziałał w epoce {epoch}!")
            break

    total_time = time.time() - start_time
    print(f"\nTrening zakończony w {total_time:.2f}s")

    print("\nGenerowanie raportów...")

    utils.plot_training_curves(history, save_path=reports_dir / "training_curves.png")

    print("\nTestowanie najlepszego modelu na zbiorze testowym...")
    utils.load_model(model, reports_dir / "best_model.pth")

    test_loss, test_acc, test_f1, test_precision, test_recall, test_preds, test_labels = utils.evaluate_model(
        model, test_loader, criterion
    )

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
        'epochs_trained': len(history['train_loss']),
        'best_val_loss': float(min(history['val_loss']))
    }
    utils.save_metrics(final_metrics, reports_dir / "metrics.json")

    print(f"\nWszystkie wyniki zapisano w: {reports_dir}")


if __name__ == "__main__":
    setup_seed(config.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['mlp', 'cnn'])
    args = parser.parse_args()
    main(args.model)
