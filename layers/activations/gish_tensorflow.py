import tensorflow as tf
from keras import layers

# Gish activation function from the paper:
# https://www.researchgate.net/publication/374231011_Gish_a_novel_activation_function_for_image_classification
class Gish(layers.Layer):

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        return x * self.alpha * tf.math.log(2 - tf.math.exp(-tf.math.exp(x * self.beta)))