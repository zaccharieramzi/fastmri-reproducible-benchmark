import tensorflow as tf
from tensorflow.keras.layers import  Layer
from tensorflow.python.ops.signal.fft_ops import fft2d, ifft2d, ifftshift, fftshift

from .masking import _mask_tf


def tf_adj_op(y, idx=0):
    x, mask = y
    x_masked = _mask_tf((x, mask))
    x_inv = tf_unmasked_adj_op(x_masked, idx=idx)
    return x_inv

def tf_unmasked_adj_op(x, idx=0):
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(tf.dtypes.cast(tf.math.reduce_prod(tf.shape(x)[1:3]), 'float32')), x.dtype)
    return scaling_norm * tf.expand_dims(fftshift(ifft2d(ifftshift(x[..., idx]))), axis=-1)

def tf_op(y, idx=0):
    x, mask = y
    x_fourier = tf_unmasked_op(x, idx=idx)
    x_masked = _mask_tf((x_fourier, mask))
    return x_masked

def tf_unmasked_op(x, idx=0):
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(tf.dtypes.cast(tf.math.reduce_prod(tf.shape(x)[1:3]), 'float32')), x.dtype)
    return tf.expand_dims(ifftshift(fft2d(fftshift(x[..., idx]))), axis=-1) / scaling_norm

def _compute_scaling_norm(x):
    image_area = tf.reduce_prod(tf.shape(x)[1:3])
    image_area = tf.cast(image_area, 'float32')
    scaling_norm = tf.sqrt(image_area)
    scaling_norm = tf.cast(scaling_norm, x.dtype)
    return scaling_norm

class FFTBase(Layer):
    def __init__(self, masked, multicoil=False, **kwargs):
        super(FFTBase, self).__init__(**kwargs)
        self.masked = masked
        self.multicoil = multicoil
        if self.multicoil:
            self.shift_axes = [2, 3]
        else:
            self.shift_axes = [1, 2]

    def get_config(self):
        config = super(FFTBase, self).get_config()
        config.update({'masked': self.masked})
        config.update({'multicoil': self.multicoil})
        return config

    def op(self, inputs):
        if self.masked:
            if self.multicoil:
                image, mask, smaps = inputs
            else:
                image, mask = inputs
        else:
            if self.multicoil:
                image, smaps = inputs
            else:
                image = inputs
        scaling_norm = _compute_scaling_norm(image)
        shifted_image = fftshift(image[..., 0], axes=self.shift_axes)
        kspace_shifted = fft2d(shifted_image)
        kspace_unnormed = ifftshift(kspace_shifted, axes=self.shift_axes)
        kspace = kspace_unnormed[..., None] / scaling_norm
        if self.masked:
            kspace = _mask_tf([kspace, mask])
        return kspace

    def adj_op(self, inputs):
        if self.masked:
            kspace, mask = inputs
            kspace = _mask_tf([kspace, mask])
        else:
            kspace = inputs
        scaling_norm = _compute_scaling_norm(kspace)
        shifted_kspace = ifftshift(kspace[..., 0], axes=self.shift_axes)
        image_shifted = ifft2d(shifted_kspace)
        image_unnormed = fftshift(image_shifted, axes=self.shift_axes)
        image = image_unnormed[..., None] * scaling_norm
        return image

class FFT(FFTBase):
    def call(self, inputs):
        return self.op(inputs)

class IFFT(FFTBase):
    def call(self, inputs):
        return self.adj_op(inputs)
