from keras.layers import Input, Lambda, Multiply, Conv2D, concatenate, Add
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.python.ops import manip_ops
from tensorflow.signal import fft2d, ifft2d

from utils import keras_psnr, keras_ssim


def to_complex(x):
    return tf.complex(x[0], x[1])

def conv2d_complex(x, n_filters, activation='relu', output_shape=None, idx=0):
    x_real = Lambda(tf.math.real, name=f'real_part_{idx}')(x)
    x_imag = Lambda(tf.math.imag, name=f'imag_part_{idx}')(x)
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
    conv_res = Lambda(to_complex, output_shape=output_shape, name=f'recomplexification_{idx}')([conv_real, conv_imag])
    return conv_res

FOURIER_SHIFT_AXES = [1, 2]
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


def tf_op(y, idx=0):
    x, mask = y
    mask_complex = tf.dtypes.cast(mask, x.dtype)
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(tf.to_float(tf.math.reduce_prod(tf.shape(x)[1:3]))), x.dtype)
    return tf.expand_dims(tf.math.multiply(mask_complex, temptf_ifft_shift(fft2d(temptf_fft_shift(x[..., idx])))), axis=-1) / scaling_norm


def tf_crop(im, crop=320):
    im_shape = tf.shape(im)
    y = im_shape[1]
    x = im_shape[2]
    startx = x // 2 - (crop // 2)
    starty = y // 2 - (crop // 2)
    im = im[:, starty:starty+crop, startx:startx+crop, :]
    return im



def invnet_crop(input_size=(640, None, 1), n_filters=32, lr=1e-3, **dummy_kwargs):
    # shapes
    mask_shape = input_size[:-1]
    # inputs and buffers
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input_simple')
    mask = Input(mask_shape, dtype='complex64', name='mask_input_simple')
    # # simple inverse
    image_res = Lambda(tf_adj_op, output_shape=input_size, name='ifft_simple')([kspace_input, mask])
    # image_res = conv2d_complex(image_res, n_filters, activation='relu', output_shape=conv_shape)
    # image_res = conv2d_complex(image_res, 1, activation='linear', output_shape=input_size)
    image_res = Lambda(tf.math.abs, name='image_module_simple')(image_res)
    image_res = Lambda(tf_crop, name='cropping')(image_res)
    model = Model(inputs=[kspace_input, mask], outputs=image_res)
    model.compile(
        optimizer=Adam(lr=lr),
        loss='mean_absolute_error',
        metrics=['mean_squared_error', keras_psnr, keras_ssim],
        # options=tf.RunOptions(report_tensor_allocations_upon_oom=True),
    )

    return model

def pdnet_crop(input_size=(640, None, 1), n_filters=32, lr=1e-3, n_primal=5, n_dual=5, n_iter=10):
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
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input')
    mask = Input(mask_shape, dtype='complex64', name='mask_input')


    primal = Lambda(lambda x: tf.concat([tf.zeros_like(x, dtype='complex64')] * n_primal, axis=-1), output_shape=primal_shape, name='buffer_primal')(kspace_input)
    dual = Lambda(lambda x: tf.concat([tf.zeros_like(kspace_input, dtype='complex64')] * n_dual, axis=-1), output_shape=dual_shape, name='buffer_dual')(kspace_input)

    for i in range(n_iter):
        # first work in kspace (dual space)
        dual_eval_exp = Lambda(tf_op, output_shape=input_size, arguments={'idx': 1}, name='fft_masked_{i}'.format(i=i+1))([primal, mask])
        update = concatenate([dual, dual_eval_exp, kspace_input], axis=-1)

        update = conv2d_complex(update, n_filters, activation='relu', output_shape=conv_shape, idx=f'{i}_1_primal')
        update = conv2d_complex(update, n_filters, activation='relu', output_shape=conv_shape, idx=f'{i}_2_primal')
        update = conv2d_complex(update, n_dual, activation='linear', output_shape=dual_shape, idx=f'{i}_linear_primal')
        dual = Add()([dual, update])

        # if only primal:
        # dual = dual_eval_exp - kspace_input


        # Then work in image space (primal space)
        primal_exp = Lambda(tf_adj_op, output_shape=input_size, name='ifft_masked_{i}'.format(i=i+1))([dual, mask])
        update = concatenate([primal, primal_exp], axis=-1)

        update = conv2d_complex(update, n_filters, activation='relu', output_shape=conv_shape, idx=f'{i}_1_dual')
        update = conv2d_complex(update, n_filters, activation='relu', output_shape=conv_shape, idx=f'{i}_2_dual')
        update = conv2d_complex(update, n_dual, activation='linear', output_shape=primal_shape, idx=f'{i}_linear_dual')
        primal = Add()([primal, update])

    image_res = Lambda(lambda x: x[..., 0:1], output_shape=input_size, name='image_getting')(primal)

    image_res = Lambda(tf.math.abs, name='image_module', output_shape=input_size)(image_res)
    image_res = Lambda(tf_crop, name='cropping', output_shape=(320, 320, 1))(image_res)
    model = Model(inputs=[kspace_input, mask], outputs=image_res)
    model.compile(
        optimizer=Adam(lr=lr),
        loss='mean_absolute_error',
        metrics=['mean_squared_error', keras_psnr, keras_ssim],
        # options=tf.RunOptions(report_tensor_allocations_upon_oom=True),
    )

    return model
