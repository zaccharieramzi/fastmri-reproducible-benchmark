import tensorflow as tf
from tensorflow.keras.layers import Lambda, concatenate, Add, Conv2D

def to_complex(x, n):
    return tf.complex(
        tf.cast(x[..., :n], dtype=tf.float32),
        tf.cast(x[..., n:], dtype=tf.float32),
    )

def to_real(x):
    return tf.concat([
        tf.math.real(x),
        tf.math.imag(x),
    ], axis=-1)

def _concatenate_real_imag(x):
    x_real = Lambda(tf.math.real)(x)
    x_imag = Lambda(tf.math.imag)(x)
    return concatenate([x_real, x_imag])

def _complex_from_half(x, n, output_shape):
    return Lambda(lambda x: to_complex(x, n), output_shape=output_shape)(x)

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
