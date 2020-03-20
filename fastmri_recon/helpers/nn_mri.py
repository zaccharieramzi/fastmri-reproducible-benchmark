"""Module containing helpers for building NN for MRI reconstruction in tf."""
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Conv2D, Layer, concatenate, Add, LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.python.ops.signal.fft_ops import fft2d, ifft2d, ifftshift, fftshift


### Keras and TensorFlow ###
FOURIER_SHIFT_AXES = [1, 2]

## complex numbers handling
def _to_complex(x):
    return tf.complex(x[0], x[1])

def _concatenate_real_imag(x):
    x_real = Lambda(tf.math.real)(x)
    x_imag = Lambda(tf.math.imag)(x)
    return concatenate([x_real, x_imag])

def _complex_from_half(x, n, output_shape):
    return Lambda(lambda x: _to_complex([x[..., :n], x[..., n:]]), output_shape=output_shape)(x)

def lrelu(x):
    return LeakyReLU(alpha=0.1)(x)

def conv2d_complex(x, n_filters, n_convs, activation='relu', output_shape=None, res=False, last_kernel_size=3):
    x_real_imag = _concatenate_real_imag(x)
    n_complex = output_shape[-1]
    for j in range(n_convs):
        x_real_imag = Conv2D(
            n_filters,
            3,
            activation=activation,
            padding='same',
            kernel_initializer='glorot_uniform',
            # kernel_regularizer=regularizers.l2(1e-6),
            # bias_regularizer=regularizers.l2(1e-6),
        )(x_real_imag)
    x_real_imag = Conv2D(
        2 * n_complex,
        last_kernel_size,
        activation='linear',
        padding='same',
        kernel_initializer='glorot_uniform',
        # kernel_regularizer=regularizers.l2(1e-6),
        # bias_regularizer=regularizers.l2(1e-6),
    )(x_real_imag)
    x_real_imag = _complex_from_half(x_real_imag, n_complex, output_shape)
    if res:
        x_final = Add()([x, x_real_imag])
    else:
        x_final = x_real_imag
    return x_final

## fourier ops definitions
def _mask_tf(x):
    k_data, mask = x
    mask = tf.expand_dims(tf.dtypes.cast(mask, k_data.dtype), axis=-1)
    masked_k_data = tf.math.multiply(mask, k_data)
    return masked_k_data

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

def tf_unmasked_fft(image):
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(tf.dtypes.cast(tf.math.reduce_prod(tf.shape(image)), 'float32')), image.dtype)
    return ifftshift(fft2d(fftshift(image))) / scaling_norm

def tf_unmasked_ifft(fft_coeffs):
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(tf.dtypes.cast(tf.math.reduce_prod(tf.shape(fft_coeffs)), 'float32')), fft_coeffs.dtype)
    return scaling_norm * fftshift(ifft2d(ifftshift(fft_coeffs)))

## Data consistency ops
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

## fastMRI helpers
def _tf_crop(im, crop=320):
    im_shape = tf.shape(im)
    y = im_shape[1]
    x = im_shape[2]
    startx = x // 2 - (crop // 2)
    starty = y // 2 - (crop // 2)
    im = im[:, starty:starty+crop, startx:startx+crop, :]
    return im

def tf_fastmri_format(image):
    image = Lambda(lambda x: _tf_crop(tf.math.abs(x)), name='cropping', output_shape=(320, 320, 1))(image)
    return image
