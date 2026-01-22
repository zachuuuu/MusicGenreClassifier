import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from clearml import Task

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import config
import performing_data.dataset as ds
from models.mlp.mlp_model import MLP
from models.cnn.cnn_model import CNN
import models.utils as utils


def get_optimizer(model, optimizer_name, lr, weight_decay):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def objective(trial, model_type, device, task):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])

    if model_type == 'mlp':
        hidden_sizes = trial.suggest_categorical('hidden_sizes', [
            (128, 64),
            (256, 128),
            (256, 128, 64),
            (512, 256, 128),
            (1024, 512, 256),
        ])

        train_loader, val_loader, test_loader, le = ds.get_mlp_dataloaders(
            config.FEATURES_FILE, config, batch_size=batch_size
        )

        model = MLP(
            input_size=config.MLP_CONFIG['input_size'],
            hidden_sizes=hidden_sizes,
            num_classes=config.NUM_CLASSES,
            dropout=dropout
        )
        epochs = config.MLP_CONFIG['epochs']
        patience = config.MLP_CONFIG['patience']

    else:

        conv_channels = trial.suggest_categorical('conv_channels', [
            [32, 64],
            [32, 64, 128],
            [64, 128, 256],
            [32, 64, 128, 256],
        ])

        fc_size = trial.suggest_categorical('fc_size', [128, 256, 512])

        train_loader, val_loader, test_loader, le = ds.get_cnn_dataloaders(
            config.SPECTROGRAMS_DIR, config.GENRES, config, batch_size=batch_size
        )

        model = CNN(
            num_classes=config.NUM_CLASSES,
            conv_channels=conv_channels,
            fc_size=fc_size,
            dropout=dropout
        )
        epochs = config.CNN_CONFIG['epochs']
        patience = config.CNN_CONFIG['patience']

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_name, learning_rate, weight_decay)

    early_stopping = utils.EarlyStopping(patience=patience, mode='min')

    best_val_loss = float('inf')

    print(f"\n--- Trial {trial.number} Start: LR={learning_rate:.5f}, Batch={batch_size}, Opt={optimizer_name} ---")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = utils.train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_f1, val_prec, val_rec, _, _ = utils.evaluate_model(
            model, val_loader, criterion, device
        )

        if task:
            logger = task.get_logger()
            logger.report_scalar('Loss', 'train', train_loss, iteration=epoch)
            logger.report_scalar('Loss', 'val', val_loss, iteration=epoch)
            logger.report_scalar('Accuracy', 'val', val_acc, iteration=epoch)
            logger.report_scalar('F1 Score', 'val', val_f1, iteration=epoch)

        trial.report(val_loss, epoch)

        print(f"Trial {trial.number} | Ep {epoch} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch}.")
            raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        early_stopping(val_loss)

        if early_stopping.early_stop:
            print(f"Trial {trial.number} Early Stopping at epoch {epoch}")
            break

    return best_val_loss


def run_optimization(model_type, n_trials=20):
    project_name = f"{config.PROJECT_NAME}_Tuning"
    task_name = f"{model_type.upper()}_Optuna_Search"

    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 60}")
    print(f"START OPTIMIZATION: {model_type.upper()}")
    print(f"Device: {device}")
    print(f"{'=' * 60}\n")

    study = optuna.create_study(
        direction='minimize',
        study_name=f"{model_type}_study",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )

    study.optimize(
        lambda trial: objective(trial, model_type, device, task),
        n_trials=n_trials
    )

    print(f"\n{'=' * 60}")
    print(f"OPTIMIZATION COMPLETED")
    print(f"{'=' * 60}")
    print(f"Best Trial ID: {study.best_trial.number}")
    print(f"Best Val Loss: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    task.close()
    return study


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna & ClearML')
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'cnn'])
    parser.add_argument('--trials', type=int, default=20, help='Number of trials')
    args = parser.parse_args()

    import random
    import numpy as np
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    run_optimization(args.model, n_trials=args.trials)

if __name__ == "__main__":
    main()