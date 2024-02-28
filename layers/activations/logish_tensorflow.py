import tensorflow as tf
from keras import layers, activations

# Logish activation function from the paper:
# https://www.sciencedirect.com/science/article/abs/pii/S0925231221009917
class Logish(layers.Layer):
    def __init__(self, alpha=1, beta=1, trainable=False, **kwargs):
        super().__init__(trainable, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs, *args):
        return inputs * self.alpha + tf.math.log(1 + activations.sigmoid(inputs * self.beta))
