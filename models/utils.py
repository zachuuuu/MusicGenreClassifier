import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score
import json
from pathlib import Path


class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, val_loader):
        super().__init__()
        self.total_epochs = total_epochs
        self.val_loader = val_loader

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        train_loss = logs.get('loss', 0)
        train_acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        val_preds = []
        val_labels = []
        for x, y in self.val_loader:
            preds = self.model(x, training=False)
            val_preds.extend(tf.argmax(preds, axis=1).numpy())
            val_labels.extend(y.numpy())

        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"Epoch {epoch_num}/{self.total_epochs} | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")



def get_predictions_and_labels(model, dataloader):
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        outputs = model(inputs, training=False)
        predicted = tf.argmax(outputs, axis=1).numpy()

        all_preds.extend(predicted)
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_training_curves(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Wykres zapisany: {save_path}")

    plt.close()


def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(10, 8))

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Accuracy (%)'}
    )

    plt.title('Confusion Matrix (%)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix zapisana: {save_path}")

    plt.close()


def save_metrics(metrics, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Metryki zapisane: {save_path}")


def print_classification_report(y_true, y_pred, class_names):
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
