import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import json
from pathlib import Path


def evaluate_model(model, dataloader, criterion):
    total_loss = 0.0
    all_preds = []
    all_labels = []

    num_samples = 0

    for inputs, labels in dataloader:
        labels = tf.cast(labels, tf.int64)
        outputs = model(inputs, training=False)
        loss = criterion(labels, outputs)

        total_loss += loss.numpy() * inputs.shape[0]
        num_samples += inputs.shape[0]

        predicted = tf.argmax(outputs, axis=1).numpy()

        all_preds.extend(predicted)
        all_labels.extend(labels.numpy())

    avg_loss = total_loss / num_samples
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1, precision, recall, np.array(all_preds), np.array(all_labels)


def train_one_epoch(model, dataloader, criterion, optimizer):
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        labels = tf.cast(labels, tf.int64)

        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = criterion(labels, outputs)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_loss += loss.numpy() * inputs.shape[0]

        predicted = tf.argmax(outputs, axis=1)
        total += labels.shape[0]
        correct += tf.reduce_sum(tf.cast(predicted == labels, tf.int32)).numpy()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


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


def save_model(model, epoch, metrics, save_path):
    weights_path = str(save_path).replace('.pth', '.weights.h5')

    model.save_weights(weights_path)

    metadata = {
        'epoch': int(epoch),
        'metrics': metrics
    }

    metadata_path = str(save_path).replace('.pth', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Model zapisany: {weights_path}")


def load_model(model, checkpoint_path):
    weights_path = str(checkpoint_path).replace('.pth', '.weights.h5')

    model.load_weights(weights_path)

    metadata_path = str(checkpoint_path).replace('.pth', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    epoch = metadata['epoch']
    metrics = metadata.get('metrics', {})

    print(f"Model wczytany z epoki {epoch}")

    return epoch, metrics


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return True

        if self.mode == 'min':
            improvement = self.best_value - current_value > self.min_delta
        else:
            improvement = current_value - self.best_value > self.min_delta

        if improvement:
            self.best_value = current_value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False