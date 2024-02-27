# Deprecated that is based on tensorflow we will switch on pytorch


from tensorflow import keras
from keras import layers, activations, regularizers
from model import Logish

import tensorflow as tf


# Homemade


class BasicDepthwiseAttentionConvolution(layers.Layer):

    def __init__(
            self,
            activation_depth="relu",
            kernel_size_depth=(3, 3),
            strides_depth=(1, 1),
            padding_depth="same",
            depth_multiplier=1,
            use_bias_depth=False,
            use_normalization=False,
            **kwargs
        ):
        super(BasicDepthwiseAttentionConvolution, self).__init__(**kwargs)

        self.kernel_size_depth  = kernel_size_depth
        self.strides_depth      = strides_depth
        self.padding_depth      = padding_depth
        self.depth_multiplier   = depth_multiplier
        self.use_bias_depth     = use_bias_depth
        self.activation_depth   = activation_depth
        self.use_normalization  = use_normalization


    def build(self, input_shape):
        self.depthwise_conv = layers.DepthwiseConv2D(
            # depthwise_initializer="ones",
            kernel_size=self.kernel_size_depth,
            padding=self.padding_depth,
            depth_multiplier=self.depth_multiplier,
            strides=self.strides_depth,
            use_bias=self.use_bias_depth,
            activation=self.activation_depth,
        )
        # self.attention_depth_weights_channels = self.add_weight(
        #     shape=(input_shape[-1] * self.depth_multiplier, input_shape[-2], input_shape[-3]),
        #     initializer="ones",
        #     trainable=True,
        # )
        self.attention_depth_weights_channels = self.add_weight(
            shape=(input_shape[-3], input_shape[-2], input_shape[-1] * self.depth_multiplier),
            # initializer="ones",
            initializer="random_normal",
            trainable=True,
        )
        self.normalization = layers.LayerNormalization(
            beta_initializer="zeros",
            gamma_initializer="ones",
            axis=0,
            epsilon=1e-6
        )
    
    def call(self, inputs, *args):
        self.__A = self.depthwise_conv(inputs)
        self.__B = self.__A + self.attention_depth_weights_channels * self.__A
        self.__C = self.__B
        if self.use_normalization:
            self.__C = self.normalization(self.__B)
        return self.__C

class DepthwiseAttentionConvolution(layers.Layer):

    def __init__(self, **kwargs):
        super(DepthwiseAttentionConvolution, self).__init__(**kwargs)
        raise NotImplementedError("This class is not implemented yet.")

    def build(self, input_shape):
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            padding="same",
            depth_multiplier=1,
            strides=(1, 1),
            use_bias=False,
            activation="relu",
            # kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)
        )
    
    def call(self, inputs, *args):
        return inputs