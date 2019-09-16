from keras.layers import Input, Lambda, Multiply, Conv2D, concatenate, Add
from keras.models import Model
from keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.python.ops import manip_ops
from tensorflow.signal import fft2d, ifft2d
import torch
from torch import nn

from torch_utils import ConvBlock, replace_values_on_mask_torch
from transforms import ifft2, fft2, center_crop, complex_abs
from utils import keras_psnr, keras_ssim


def to_complex(x):
    return tf.complex(x[0], x[1])

def concatenate_real_imag(x):
    x_real = Lambda(tf.math.real)(x)
    x_imag = Lambda(tf.math.imag)(x)
    return concatenate([x_real, x_imag])

def complex_from_half(x, n, output_shape):
    return Lambda(lambda x: to_complex([x[..., :n], x[..., n:]]), output_shape=output_shape)(x)

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



def invnet_crop(input_size=(640, None, 1), n_filters=32, lr=1e-3, **dummy_kwargs):
    # shapes
    mask_shape = input_size[:-1]
    # inputs and buffers
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input_simple')
    mask = Input(mask_shape, dtype='complex64', name='mask_input_simple')
    # # simple inverse
    image_res = Lambda(tf_adj_op, output_shape=input_size, name='ifft_simple')([kspace_input, mask])
    image_res = Lambda(tf.math.abs, name='image_module_simple')(image_res)
    image_res = Lambda(tf_crop, name='cropping')(image_res)
    model = Model(inputs=[kspace_input, mask], outputs=image_res)
    model.compile(
        optimizer=Adam(lr=lr),
        loss='mean_absolute_error',
        metrics=['mean_squared_error', keras_psnr, keras_ssim],
    )

    return model

def pdnet_crop(input_size=(640, None, 1), n_filters=32, lr=1e-3, n_primal=5, n_dual=5, n_iter=10, res_connection=False, primal_only=False):
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
    if not primal_only:
        dual = Lambda(lambda x: tf.concat([tf.zeros_like(x, dtype='complex64')] * n_dual, axis=-1), output_shape=dual_shape, name='buffer_dual')(kspace_input)

    for i in range(n_iter):
        # first work in kspace (dual space)
        dual_eval_exp = Lambda(tf_op, output_shape=input_size, arguments={'idx': 1}, name='fft_masked_{i}'.format(i=i+1))([primal, mask])
        if primal_only:
            dual = Lambda(lambda x: x[0] - x[1], output_shape=input_size, name='dual_residual_{i}'.format(i=i+1))([dual_eval_exp, kspace_input])
        else:
            update = concatenate([dual, dual_eval_exp, kspace_input], axis=-1)
            update = concatenate_real_imag(update)

            update = Conv2D(
                n_filters,
                3,
                activation='relu',
                padding='same',
                kernel_initializer='glorot_uniform',
            )(update)
            update = Conv2D(
                n_filters,
                3,
                activation='relu',
                padding='same',
                kernel_initializer='glorot_uniform',
            )(update)
            update = Conv2D(
                2 * n_dual,
                3,
                activation='linear',
                padding='same',
                kernel_initializer='glorot_uniform',
            )(update)
            update = complex_from_half(update, n_dual, dual_shape)
            if res_connection:
                to_add = [update, dual_eval_exp]
            else:
                to_add = [dual, update]
            dual = Add()(to_add)


        # Then work in image space (primal space)
        primal_exp = Lambda(tf_adj_op, output_shape=input_size, name='ifft_masked_{i}'.format(i=i+1))([dual, mask])
        update = concatenate([primal, primal_exp], axis=-1)
        update = concatenate_real_imag(update)

        update = Conv2D(
            n_filters,
            3,
            activation='relu',
            padding='same',
            kernel_initializer='glorot_uniform',
        )(update)
        update = Conv2D(
            n_filters,
            3,
            activation='relu',
            padding='same',
            kernel_initializer='glorot_uniform',
        )(update)
        update = Conv2D(
            2 * n_primal,
            3,
            activation='linear',
            padding='same',
            kernel_initializer='glorot_uniform',
        )(update)
        update = complex_from_half(update, n_primal, primal_shape)
        if res_connection:
            to_add = [update, primal_exp]
        else:
            to_add = [primal, update]
        primal = Add()(to_add)

    image_res = Lambda(lambda x: x[..., 0:1], output_shape=input_size, name='image_getting')(primal)

    image_res = Lambda(tf.math.abs, name='image_module', output_shape=input_size)(image_res)
    image_res = Lambda(tf_crop, name='cropping', output_shape=(320, 320, 1))(image_res)
    model = Model(inputs=[kspace_input, mask], outputs=image_res)
    model.compile(
        optimizer=SGD(lr=lr, clipnorm=1., momentum=0.9),
        loss='mean_absolute_error',
        metrics=['mean_squared_error', keras_psnr, keras_ssim],
    )

    return model


class PDNetCrop(torch.nn.Module):
    def __init__(self, n_filters=32, n_primal=5, n_dual=5, n_iter=10, primal_only=False):
        super(PDNetCrop, self).__init__()
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.n_iter = n_iter
        self.n_filters = n_filters
        self.primal_only = primal_only

        self.primal_conv_layers = nn.ModuleList([ConvBlock(3, n_filters, 2 * (n_primal + 1), 2 * n_primal) for _ in range(n_iter)])
        if not self.primal_only:
            self.dual_conv_layers = nn.ModuleList([ConvBlock(3, n_filters, 2 * (n_dual + 2), 2 * n_dual) for _ in range(n_iter)])


    def forward(self, kspace, mask):
        mask = mask[..., None]
        mask = mask.expand_as(kspace).float()
        primal = torch.stack([torch.zeros_like(kspace)] * self.n_primal, dim=-1)
        if not self.primal_only:
            dual = torch.stack([torch.zeros_like(kspace)] * self.n_dual, dim=-1)

        for i, primal_conv_layer in enumerate(self.primal_conv_layers):
            dual_eval_exp = fft2(primal[..., 1])
            dual_eval_exp = dual_eval_exp * mask
            if self.primal_only:
                dual = dual_eval_exp - kspace
            else:
                update = torch.cat([dual[:, :, :, 0], dual[:, :, :, 1], dual_eval_exp, kspace], axis=-1)
                update = update.permute(0, 3, 1, 2)
                update = self.dual_conv_layers[i](update)
                update = update.permute(0, 2, 3, 1)
                update = torch.stack([update[..., :self.n_dual], update[..., self.n_dual:]], dim=-1)
                update = update.permute(0, 1, 2, 4, 3)
                dual = dual + update

            primal_exp = ifft2(mask * dual[..., 0])
            update = torch.cat([primal[:, :, :, 0], primal[:, :, :, 1], primal_exp], axis=-1)
            update = update.permute(0, 3, 1, 2)
            update = primal_conv_layer(update)
            update = update.permute(0, 2, 3, 1)
            update = torch.stack([update[..., :self.n_primal], update[..., self.n_primal:]], dim=-1)
            update = update.permute(0, 1, 2, 4, 3)
            primal = primal + update


        image = primal[..., 0]
        # equivalent of taking the module of the image
        image = complex_abs(image)
        image = center_crop(image, (320, 320))
        return image
