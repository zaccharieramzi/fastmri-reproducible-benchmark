import tensorflow as tf
from tensorflow.keras.layers import Layer, Lambda, Add

from .masking import _mask_tf


class MultiplyScalar(Layer):
    def __init__(self, **kwargs):
        super(MultiplyScalar, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.sample_weight = self.add_weight(
            name='sample_weight',
            shape=(1,),
            initializer='ones',
            trainable=True,
        )
        super(MultiplyScalar, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.cast(self.sample_weight, tf.complex64) * x

    def compute_output_shape(self, input_shape):
        return input_shape

def _replace_values_on_mask(x):
    # TODO: check in multicoil case
    cnn_fft, kspace_input, mask = x
    anti_mask = tf.expand_dims(tf.dtypes.cast(1.0 - mask, cnn_fft.dtype), axis=-1)
    replace_cnn_fft = tf.math.multiply(anti_mask, cnn_fft) + kspace_input
    return replace_cnn_fft

def enforce_kspace_data_consistency(kspace, kspace_input, mask, input_size, multiply_scalar=None, noiseless=True):
    if noiseless:
        data_consistent_kspace = Lambda(_replace_values_on_mask, output_shape=input_size)([kspace, kspace_input, mask])
    else:
        if multiply_scalar is None:
            multiply_scalar = MultiplyScalar()
        kspace_masked = Lambda(lambda x: -_mask_tf(x), output_shape=input_size)([kspace, mask])
        data_consistent_kspace = Add()([kspace_input, kspace_masked])
        data_consistent_kspace = multiply_scalar(data_consistent_kspace)
        data_consistent_kspace = Add()([data_consistent_kspace, kspace])
    return data_consistent_kspace
