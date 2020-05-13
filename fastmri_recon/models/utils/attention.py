import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Activation, LeakyReLU, PReLU, Dense, Conv1D

def _conv1d(conv_layer):
    def _conv1d_fun(inputs):
        conv = conv_layer(inputs[:, None, :])
        conv = conv[:, 0, :]
        return conv
    return _conv1d_fun

class ChannelAttentionBlock(Layer):
    def __init__(self, reduction_factor=4, dense=False, activation='relu', **kwargs):
        super(ChannelAttentionBlock, self).__init__(**kwargs)
        self.reduction_factor = reduction_factor
        self.dense = dense
        self.ga_pooling = GlobalAveragePooling2D()
        self.activation_str = activation
        if self.activation_str == 'lrelu':
            self.activation = LeakyReLU(0.1)
        elif self.activation_str == 'prelu':
            self.activation = PReLU(shared_axes=[])
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        n_channels = input_shape[-1]
        n_reduced = n_channels // self.reduction_factor
        if self.dense:
            self.squeeze = Dense(n_reduced)
            self.expand = Dense(n_channels)
        else:
            self.squeeze = _conv1d(Conv1D(n_reduced, 1, activation='linear'))
            self.expand = _conv1d(Conv1D(n_channels, 1, activation='linear'))
        super(ChannelAttentionBlock, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        global_average = self.ga_pooling(inputs)
        attentions = self.squeeze_and_expand(global_average)
        scaled_inputs = inputs * attentions[:, None, None, :]
        return scaled_inputs

    def squeeze_and_expand(self, inputs):
        squeezed = self.squeeze(inputs)
        squeezed_excited = self.activation(squeezed)
        expanded = self.expand(squeezed_excited)
        expanded_weights = tf.nn.sigmoid(expanded)
        return expanded_weights

    def get_config(self):
        config = super(ChannelAttentionBlock, self).get_config()
        config.update({
            'dense': self.dense,
            'reduction_factor': self.reduction_factor,
            'activation': self.activation_str,
        })
        return config
