import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CNN(keras.Model):
    def __init__(self, num_classes, conv_channels=None, fc_size=256, dropout=0.3):
        super(CNN, self).__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.conv_layers = []

        for out_channels in conv_channels:
            conv_block = keras.Sequential([
                layers.Conv2D(
                    out_channels,
                    kernel_size=3,
                    padding='same',
                ),
                layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                layers.ReLU(),
                layers.MaxPooling2D(pool_size=2, strides=2)
            ])
            self.conv_layers.append(conv_block)

        self.adaptive_pool = layers.GlobalAveragePooling2D()

        self.fc1 = layers.Dense(
            fc_size
        )
        self.relu = layers.ReLU()
        self.dropout_fc = layers.Dropout(dropout)
        self.fc2 = layers.Dense(
            num_classes, activation='softmax'
        )

    def call(self, x, training=False):
        for conv_layer in self.conv_layers:
            x = conv_layer(x, training=training)

        x = self.adaptive_pool(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x, training=training)
        x = self.fc2(x)

        return x
