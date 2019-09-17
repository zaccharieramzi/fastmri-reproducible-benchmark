from keras.layers import Input, Lambda, Conv2D, Add, Layer
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import torch
from torch import nn

from ..helpers.nn_mri_helpers import tf_adj_op, tf_op, concatenate_real_imag, complex_from_half, tf_crop, tf_unmasked_adj_op, tf_unmasked_op, replace_values_on_mask_torch, MultiplyScalar, replace_values_on_mask
from ..helpers.utils import keras_psnr, keras_ssim
from ..helpers.torch_utils import ConvBlock
from ..helpers.transforms import ifft2, fft2, center_crop, complex_abs


def cascade_net(input_size=(640, None, 1), n_cascade=5, n_convs=5, n_filters=16, noiseless=True, lr=1e-3):
    mask_shape = input_size[:-1]
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input')
    mask = Input(mask_shape, dtype='complex64', name='mask_input')

    zero_filled = Lambda(tf_unmasked_adj_op, output_shape=input_size, name='ifft_simple')(kspace_input)


    image = zero_filled
    multiply_scalar = MultiplyScalar()
    for i in range(n_cascade):
        # residual convolution
        res_image = concatenate_real_imag(image)
        for j in range(n_convs):
            res_image = Conv2D(
                n_filters,
                3,
                activation='relu',
                padding='same',
                kernel_initializer='glorot_uniform',
            )(res_image)
        res_image = Conv2D(
            2,
            3,
            activation='linear',
            padding='same',
            kernel_initializer='glorot_uniform',
        )(res_image)
        res_image = complex_from_half(res_image, 1, input_size)
        image = Add(name='res_connex_{i}'.format(i=i+1))([image, res_image])
        # data consistency layer
        if noiseless:
            cnn_fft = Lambda(tf_unmasked_op, output_shape=input_size, name='fft_simple_{i}'.format(i=i+1))(image)
            data_consistency_fourier = Lambda(replace_values_on_mask, output_shape=input_size, name='fft_repl_{i}'.format(i=i+1))([cnn_fft, kspace_input, mask])
            image = Lambda(tf_unmasked_adj_op, output_shape=input_size, name='ifft_simple_{i}'.format(i=i+1))(data_consistency_fourier)
        else:
            cnn_fft = Lambda(tf_op, output_shape=input_size, name='fft_masked_{i}'.format(i=i+1))([image, mask])
            cnn_fft = Lambda(lambda x: -x, output_shape=input_size)(cnn_fft)
            data_consistency_fourier = Add(name='data_consist_fft_{i}'.format(i=i+1))([kspace_input, cnn_fft])
            data_consistency_image = Lambda(tf_adj_op, output_shape=input_size, name='ifft_masked_{i}'.format(i=i+1))([data_consistency_fourier, mask])
            data_consistency_image = multiply_scalar(data_consistency_image)
            image = Add(name='data_consist_enf_{i}'.format(i=i+1))([image, data_consistency_image])

    # module and crop of image
    image = Lambda(tf.math.abs, name='image_module', output_shape=input_size)(image)
    image = Lambda(tf_crop, name='cropping', output_shape=(320, 320, 1))(image)
    model = Model(inputs=[kspace_input, mask], outputs=image)

    model.compile(
        optimizer=Adam(lr=lr, clipnorm=1.),
        loss='mean_absolute_error',
        metrics=['mean_squared_error', keras_psnr, keras_ssim],
    )

    return model


class CascadeNet(torch.nn.Module):
    def __init__(self, n_cascade=5, n_convs=5, n_filters=16):
        super(CascadeNet, self).__init__()
        self.n_cascade = n_cascade
        self.n_convs = n_convs
        self.n_filters = n_filters

        self.conv_layers = nn.ModuleList([ConvBlock(n_convs, n_filters) for _ in range(n_cascade)])

    def forward(self, kspace, mask):
        zero_filled = ifft2(kspace)
        image = zero_filled
        # this because pytorch doesn't support NHWC
        for i, conv_layer in enumerate(self.conv_layers):
            # residual convolution
            res_image = image
            res_image = res_image.permute(0, 3, 1, 2)
            res_image = conv_layer(res_image)
            res_image = res_image.permute(0, 2, 3, 1)
            image = image + res_image
            # data consistency layer
            cnn_fft = fft2(image)
            data_consistency_fourier = replace_values_on_mask_torch(cnn_fft, kspace, mask)
            image = ifft2(data_consistency_fourier)

        # equivalent of taking the module of the image
        image = complex_abs(image)
        image = center_crop(image, (320, 320))
        return image
