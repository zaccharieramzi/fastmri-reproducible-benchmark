"""Module containing helpers for building NN for MRI reconstruction in pytorch and keras."""
from keras.layers import Lambda, Conv2D, Layer, concatenate, Add
import tensorflow as tf
from tensorflow.signal import fft2d, ifft2d
from tensorflow.python.ops import manip_ops


### Keras and TensorFlow ###
FOURIER_SHIFT_AXES = [1, 2]

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

def replace_values_on_mask(x):
    cnn_fft, kspace_input, mask = x
    anti_mask = tf.expand_dims(tf.dtypes.cast(1.0 - mask, cnn_fft.dtype), axis=-1)
    replace_cnn_fft = tf.math.multiply(anti_mask, cnn_fft) + kspace_input
    return replace_cnn_fft

def mask_tf(x):
    k_data, mask = x
    mask = tf.expand_dims(tf.dtypes.cast(mask, k_data.dtype), axis=-1)
    masked_k_data = tf.math.multiply(mask, k_data)
    return masked_k_data

def _to_complex(x):
    return tf.complex(x[0], x[1])

def _concatenate_real_imag(x):
    x_real = Lambda(tf.math.real)(x)
    x_imag = Lambda(tf.math.imag)(x)
    return concatenate([x_real, x_imag])

def _complex_from_half(x, n, output_shape):
    return Lambda(lambda x: _to_complex([x[..., :n], x[..., n:]]), output_shape=output_shape)(x)

def conv2d_complex(x, n_filters, n_convs, activation='relu', output_shape=None, res=False):
    x_real_imag = _concatenate_real_imag(x)
    n_complex = output_shape[-1]
    for j in range(n_convs):
        x_real_imag = Conv2D(
            n_filters,
            3,
            activation=activation,
            padding='same',
            kernel_initializer='glorot_uniform',
        )(x_real_imag)
    x_real_imag = Conv2D(
        2 * n_complex,
        3,
        activation='linear',
        padding='same',
        kernel_initializer='glorot_uniform',
    )(x_real_imag)
    x_real_imag = _complex_from_half(x_real_imag, n_complex, output_shape)
    if res:
        x_final = Add()([x, x_real_imag])
    else:
        x_final = x_real_imag
    return x_final

def temptf_fft_shift(x):
    # taken from https://github.com/tensorflow/tensorflow/pull/27075/files
    shift = [tf.shape(x)[ax] // 2 for ax in FOURIER_SHIFT_AXES]
    return manip_ops.roll(x, shift, FOURIER_SHIFT_AXES)


def temptf_ifft_shift(x):
    # taken from https://github.com/tensorflow/tensorflow/pull/27075/files
    shift = [-tf.cast(tf.shape(x)[ax] // 2, tf.int32) for ax in FOURIER_SHIFT_AXES]
    return manip_ops.roll(x, shift, FOURIER_SHIFT_AXES)

def tf_adj_op(y, idx=0):
    x, mask = y
    mask_complex = tf.dtypes.cast(mask, x.dtype)
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(tf.to_float(tf.math.reduce_prod(tf.shape(x)[1:3]))), x.dtype)
    return scaling_norm * tf.expand_dims(temptf_fft_shift(ifft2d(temptf_ifft_shift(tf.math.multiply(mask_complex, x[..., idx])))), axis=-1)

def tf_unmasked_adj_op(x, idx=0):
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(tf.to_float(tf.math.reduce_prod(tf.shape(x)[1:3]))), x.dtype)
    return scaling_norm * tf.expand_dims(temptf_fft_shift(ifft2d(temptf_ifft_shift(x[..., idx]))), axis=-1)


def tf_op(y, idx=0):
    x, mask = y
    mask_complex = tf.dtypes.cast(mask, x.dtype)
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(tf.to_float(tf.math.reduce_prod(tf.shape(x)[1:3]))), x.dtype)
    return tf.expand_dims(tf.math.multiply(mask_complex, temptf_ifft_shift(fft2d(temptf_fft_shift(x[..., idx])))), axis=-1) / scaling_norm


def tf_unmasked_op(x, idx=0):
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(tf.to_float(tf.math.reduce_prod(tf.shape(x)[1:3]))), x.dtype)
    return tf.expand_dims(temptf_ifft_shift(fft2d(temptf_fft_shift(x[..., idx]))), axis=-1) / scaling_norm


def tf_crop(im, crop=320):
    im_shape = tf.shape(im)
    y = im_shape[1]
    x = im_shape[2]
    startx = x // 2 - (crop // 2)
    starty = y // 2 - (crop // 2)
    im = im[:, starty:starty+crop, startx:startx+crop, :]
    return im


### PyTorch ###
def replace_values_on_mask_torch(cnn_fft, kspace, mask):
    mask = mask[..., None]
    mask = mask.expand_as(kspace).float()
    anti_mask = 1.0 - mask
    replace_cnn_fft = anti_mask * cnn_fft + kspace
    return replace_cnn_fft
