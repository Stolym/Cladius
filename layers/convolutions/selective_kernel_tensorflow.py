import tensorflow as tf
from keras import layers, models

class SKLayer(tf.keras.layers.Layer):
    def __init__(self, channels, M=2, G=32, r=16, L=32):
        super(SKLayer, self).__init__()
        self.M = M
        self.convs = []
        for i in range(M):
            self.convs.append(models.Sequential([
                layers.Conv2D(channels, 3 + 2*i, padding='same', groups=G, activation='relu'),
                layers.BatchNormalization()
            ]))
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = models.Sequential([
            layers.Dense(L, activation='relu'),
            layers.Dense(channels * M, activation='softmax')
        ])
        self.channels = channels

    def call(self, inputs):
        outputs = []
        for conv in self.convs:
            outputs.append(conv(inputs))
        outputs = tf.concat(outputs, axis=-1)
        se_weight = self.gap(outputs)
        se_weight = self.fc(se_weight)
        se_weight = tf.reshape(se_weight, [-1, 1, 1, self.M, self.channels])
        outputs = tf.reshape(outputs, [-1, inputs.shape[1], inputs.shape[2], self.M, self.channels])
        outputs = tf.reduce_sum(outputs * se_weight, axis=-2)
        return outputs
