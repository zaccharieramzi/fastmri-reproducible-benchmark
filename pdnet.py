from keras.layers import Input, Lambda, Multiply, Conv2D, concatenate, Add
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.signal import fft2d, ifft2d

from utils import keras_psnr, keras_ssim


def to_complex(x):
    return tf.complex(x[0], x[1])

def conv2d_complex(x, n_filters, activation='relu', output_shape=None):
    x_real = Lambda(tf.math.real)(x)
    x_imag = Lambda(tf.math.imag)(x)
    conv_real = Conv2D(
        n_filters,
        3,
        activation=activation,
        padding='same',
        kernel_initializer='he_normal',
        use_bias=False,
    )(x_real)
    conv_imag = Conv2D(
        n_filters,
        3,
        activation=activation,
        padding='same',
        kernel_initializer='he_normal',
        use_bias=False,
    )(x_imag)
    conv_res = Lambda(to_complex, output_shape=output_shape)([conv_real, conv_imag])
    return conv_res

def tf_ifft(x):
    return tf.expand_dims(ifft2d(x[..., 0]), axis=-1)

def tf_ifft_masked(y, idx=0):
    x, mask = y
    return tf.expand_dims(ifft2d(tf.math.multiply(mask, x[..., idx])), axis=-1)

def tf_fft(x):
    return tf.expand_dims(fft2d(x[..., 0]), axis=-1)

def tf_fft_masked(y, idx=0):
    x, mask = y
    return tf.expand_dims(tf.math.multiply(mask, fft2d(x[..., idx])), axis=-1)


def invnet(input_size=(640, None, 1), n_filters=32, lr=1e-3):
    # shapes
    mask_shape = input_size[:-1]
    # inputs and buffers
    kspace_input = Input(input_size, dtype='complex64')
    mask = Input(mask_shape, dtype='complex64')
    # # simple inverse
    image_res = Lambda(tf_ifft, output_shape=input_size, name='ifft')(kspace_input)
    # image_res = conv2d_complex(image_res, n_filters, activation='relu', output_shape=conv_shape)
    # image_res = conv2d_complex(image_res, 1, activation='linear', output_shape=input_size)
    image_res = Lambda(tf.math.abs)(image_res)
    model = Model(inputs=[kspace_input, mask], outputs=image_res)
    model.compile(
        optimizer=Adam(lr=lr),
        loss='mean_absolute_error',
        metrics=['mean_squared_error', keras_psnr, keras_ssim],
        # options=tf.RunOptions(report_tensor_allocations_upon_oom=True),
    )

    return model

def pdnet(input_size=(640, None, 1), n_filters=32, lr=1e-3, n_primal=5, n_dual=5, n_iter=10):
    # shapes
    mask_shape = input_size[:-1]
    primal_shape = list(input_size)
    primal_shape[-1] = n_primal
    primal_shape = tuple(primal_shape)
    dual_shape = list(input_size)
    dual_shape[-1] = n_dual
    dual_shape = tuple(dual_shape)
    conv_shape = list(input_size)
    conv_shape[-1] = n_filters
    conv_shape = tuple(conv_shape)

    # inputs and buffers
    kspace_input = Input(input_size, dtype='complex64')
    mask = Input(mask_shape, dtype='complex64')


    primal = Lambda(lambda x: tf.concat([tf.zeros_like(x, dtype='complex64')] * n_primal, axis=-1), output_shape=primal_shape)(kspace_input)
    dual = Lambda(lambda x: tf.concat([tf.zeros_like(kspace_input, dtype='complex64')] * n_dual, axis=-1), output_shape=dual_shape)(kspace_input)

    for i in range(n_iter):
        # first work in kspace (dual space)
        dual_eval_exp = Lambda(tf_fft_masked, output_shape=input_size, arguments={'idx': 1}, name='fft_masked_{i}'.format(i=i+1))([primal, mask])
        update = concatenate([dual, dual_eval_exp, kspace_input], axis=-1)

        update = conv2d_complex(update, n_filters, activation='relu', output_shape=conv_shape)
        update = conv2d_complex(update, n_filters, activation='relu', output_shape=conv_shape)
        update = conv2d_complex(update, n_dual, activation='linear', output_shape=dual_shape)
        dual = Add()([dual, update])

        # if only primal:
        # dual = dual_eval_exp - kspace_input


        # Then work in image space (primal space)
        primal_exp = Lambda(tf_ifft_masked, output_shape=input_size, name='ifft_masked_{i}'.format(i=i+1))([dual, mask])
        update = concatenate([primal, primal_exp], axis=-1)

        update = conv2d_complex(update, n_filters, activation='relu', output_shape=conv_shape)
        update = conv2d_complex(update, n_filters, activation='relu', output_shape=conv_shape)
        update = conv2d_complex(update, n_dual, activation='linear', output_shape=primal_shape)
        primal = Add()([primal, update])

    image_res = Lambda(lambda x: x[..., 0:1], output_shape=input_size)(primal)

    image_res = Lambda(tf.math.abs)(image_res)
    model = Model(inputs=[kspace_input, mask], outputs=image_res)
    model.compile(
        optimizer=Adam(lr=lr),
        loss='mean_absolute_error',
        metrics=['mean_squared_error', keras_psnr, keras_ssim],
        # options=tf.RunOptions(report_tensor_allocations_upon_oom=True),
    )

    return model
