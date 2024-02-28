import tensorflow as tf
from keras import layers, models

class SELayer(layers.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.pool = layers.GlobalAveragePooling2D()
        self.fc = models.Sequential([
            layers.Dense(channel // reduction, activation='relu'),
            layers.Dense(channel, activation='sigmoid')
        ])

    def call(self, inputs):
        x = self.pool(inputs)
        x = self.fc(x)
        x = tf.reshape(x, [-1, 1, 1, inputs.shape[-1]])
        return inputs * x
