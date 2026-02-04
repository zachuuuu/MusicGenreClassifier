import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MLP(keras.Model):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout):
        super(MLP, self).__init__()

        model_layers = []

        model_layers.append(layers.Dense(
            hidden_sizes[0],
            input_shape=(input_size,),
        ))
        model_layers.append(layers.BatchNormalization())
        model_layers.append(layers.ReLU())
        model_layers.append(layers.Dropout(dropout))

        for i in range(len(hidden_sizes) - 1):
            model_layers.append(layers.Dense(
                hidden_sizes[i + 1],
            ))
            model_layers.append(layers.BatchNormalization())
            model_layers.append(layers.ReLU())
            model_layers.append(layers.Dropout(dropout))

        model_layers.append(layers.Dense(
            num_classes, activation='softmax'
        ))

        self.network = keras.Sequential(model_layers)

    def call(self, x, training=False):
        return self.network(x, training=training)