import tensorflow as tf

from keras import layers, activations

# Homemade, the paper will be released soon
class MagicAutomateTrendInteract(layers.Layer):
    def __init__(self, **kwargs):
        super(MagicAutomateTrendInteract, self).__init__(**kwargs)
        self.params = self.add_weight(name='params', shape=(12,), initializer=tf.initializers.RandomNormal(0.1, 0.001), trainable=True)
        self.params_bias = self.add_weight(name='params_bias', shape=(6,), initializer=tf.initializers.Zeros(), trainable=True)

    def call(self, inputs, *args):
        alpha, alpham, beta, betam, gamma, gammam, gammad, delta, deltam, epsilon, epsilonm, zeta = tf.unstack(self.params)
        balpha, bbeta, bgamma, bdelta, bepsilon, bzeta = tf.unstack(self.params_bias)
        
        gelu_part = alpham * tf.multiply(inputs, tf.sigmoid(alpha * (inputs * 1.702))) + balpha
        soft_part = betam * tf.keras.activations.softmax(beta * inputs) + bbeta 
        daa_part = gamma * inputs + gammam * tf.math.exp(gammad * inputs) + bgamma
        naaa_part = deltam * tf.tanh(delta * (2 * inputs)) + bdelta
        paaa_part = epsilonm * tf.math.log(1 + 0.5 * tf.abs(epsilon * inputs)) + bepsilon
        aaa_part = tf.where(inputs < 0, naaa_part, paaa_part)
        linear_part = zeta * inputs + bzeta
        combined_activation = (gelu_part * delta + soft_part * betam + daa_part * gamma + aaa_part * epsilon + linear_part * zeta) / (delta + betam + gamma + epsilon + zeta)
        
        return combined_activation

# Homemade, the paper will be released soon
class MagicAutomateTrendInteract2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MagicAutomateTrendInteract2, self).__init__(**kwargs)
        self.params = self.add_weight(name='params', shape=(12,), initializer=tf.initializers.RandomUniform(0.1, 0.01), trainable=True)
        self.params_bias = self.add_weight(name='params_bias', shape=(6,), initializer=tf.initializers.Zeros(), trainable=True)

    def call(self, inputs, *args):
        alpha, alpham, beta, betam, gamma, gammam, gammad, delta, deltam, epsilon, epsilonm, zeta = tf.unstack(self.params)
        balpha, bbeta, bgamma, bdelta, bepsilon, bzeta = tf.unstack(self.params_bias)
        
        gelu_part = alpham * tf.multiply(inputs, tf.sigmoid(alpha * (inputs * 1.702))) + balpha
        soft_part = betam * tf.keras.activations.softmax(beta * inputs) + bbeta 
        daa_part = gamma * inputs + gammam * tf.math.exp(gammad * inputs) + bgamma
        naaa_part = deltam * tf.tanh(delta * (2 * inputs)) + bdelta
        paaa_part = epsilonm * tf.math.log(1 + 0.5 * tf.abs(epsilon * inputs)) + bepsilon
        aaa_part = tf.where(inputs < 0, naaa_part, paaa_part)
        linear_part = zeta * inputs + bzeta

        combined_activation = (gelu_part + soft_part + daa_part + aaa_part + linear_part)
        
        return combined_activation

