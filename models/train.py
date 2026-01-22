import argparse
import sys
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import config
import performing_data.dataset as ds
from models.mlp.mlp_model import MLP
from models.cnn.cnn_model import CNN
import models.utils as utils

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model, cfg):
    opt_name = cfg['optimizer'].lower()

    if opt_name == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay']
        )
    elif opt_name == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay'],
            momentum=cfg.get('momentum')
        )
    elif opt_name == 'rmsprop':
        return optim.RMSprop(
            model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay']
        )
    else:
        raise ValueError(f"Nieznany optymalizator: {opt_name}. Wybierz 'adam', 'sgd' lub 'rmsprop'.")


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
    else:
        raise ValueError("Model musi być 'mlp' lub 'cnn'")

    reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Używane urządzenie: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, cfg)
    print(f"Optimizer: {cfg['optimizer'].upper()} | LR: {cfg['learning_rate']} | Weight Decay: {cfg['weight_decay']}")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    early_stopping = utils.EarlyStopping(patience=cfg['patience'], mode='min')

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\nStart treningu ({cfg['epochs']} epok)...")
    start_time = time.time()

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss, train_acc = utils.train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_f1, val_precision, val_recall, _, _ = utils.evaluate_model(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{cfg['epochs']} | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if early_stopping(val_loss):
            utils.save_model(
                model, optimizer, epoch,
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
    utils.load_model(model, optimizer, reports_dir / "best_model.pth", device)

    test_loss, test_acc, test_f1, test_precision, test_recall, test_preds, test_labels = utils.evaluate_model(
        model, test_loader, criterion, device
    )

    print(f"Test Result -> Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | "
          f"F1: {test_f1:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")

    utils.print_classification_report(test_labels, test_preds, class_names)

    cm = confusion_matrix(test_labels, test_preds)
    utils.plot_confusion_matrix(cm, class_names, save_path=reports_dir / "confusion_matrix.png")

    final_metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'training_time': total_time,
        'epochs_trained': len(history['train_loss']),
        'best_val_loss': min(history['val_loss'])
    }
    utils.save_metrics(final_metrics, reports_dir / "metrics.json")

    print(f"\nWszystkie wyniki zapisano w: {reports_dir}")


if __name__ == "__main__":
    setup_seed(config.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['mlp', 'cnn'])
    args = parser.parse_args()
    main(args.model)