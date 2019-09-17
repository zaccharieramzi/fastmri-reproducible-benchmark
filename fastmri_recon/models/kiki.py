from keras.layers import Input, Lambda, Add
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import torch
from torch import nn

from ..helpers.nn_mri_helpers import tf_crop, tf_unmasked_adj_op, tf_unmasked_op, MultiplyScalar, replace_values_on_mask, replace_values_on_mask_torch, mask_tf, conv2d_complex
from ..helpers.torch_utils import ConvBlock
from ..helpers.utils import keras_psnr, keras_ssim
from ..helpers.transforms import ifft2, fft2, center_crop, complex_abs


def kiki_net(input_size=(640, None, 1), n_cascade=5, n_convs=5, n_filters=16, noiseless=True, lr=1e-3):
    mask_shape = input_size[:-1]
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input')
    mask = Input(mask_shape, dtype='complex64', name='mask_input')

    zero_filled = Lambda(tf_unmasked_adj_op, output_shape=input_size, name='ifft_simple')(kspace_input)

    image = zero_filled
    multiply_scalar = MultiplyScalar()
    for i in range(n_cascade):
        # residual convolution (I-net)
        image = conv2d_complex(image, n_filters, n_convs, output_shape=input_size, res=True)
        # data consistency layer
        cnn_fft = Lambda(tf_unmasked_op, output_shape=input_size, name='fft_simple_{i}'.format(i=i+1))(image)
        if noiseless:
            data_consistency_fourier = Lambda(replace_values_on_mask, output_shape=input_size, name='fft_repl_{i}'.format(i=i+1))([cnn_fft, kspace_input, mask])
        else:
            cnn_fft_masked = Lambda(mask_tf, output_shape=input_size)([cnn_fft, mask])
            cnn_fft_masked = Lambda(lambda x: -x, output_shape=input_size)(cnn_fft_masked)
            data_consistency_fourier = Add(name='data_consist_fft_{i}'.format(i=i+1))([kspace_input, cnn_fft_masked])
            data_consistency_fourier = multiply_scalar(data_consistency_fourier)
            data_consistency_fourier = Add()([data_consistency_fourier, cnn_fft])
        # K-net
        data_consistency_fourier = conv2d_complex(data_consistency_fourier, n_filters, n_convs, output_shape=input_size, res=False)

        image = Lambda(tf_unmasked_adj_op, output_shape=input_size, name='ifft_simple_{i}'.format(i=i+1))(data_consistency_fourier)

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


class KikiNet(torch.nn.Module):
    def __init__(self, n_cascade=5, n_convs=5, n_filters=16):
        super(KikiNet, self).__init__()

        self.n_cascade = n_cascade
        self.n_convs = n_convs
        self.n_filters = n_filters

        self.i_conv_layers = nn.ModuleList([ConvBlock(n_convs, n_filters) for _ in range(n_cascade)])
        self.k_conv_layers = nn.ModuleList([ConvBlock(n_convs, n_filters) for _ in range(n_cascade)])

    def forward(self, kspace, mask):
        zero_filled = ifft2(kspace)
        image = zero_filled
        # this because pytorch doesn't support NHWC
        for i, (i_conv_layer, k_conv_layer) in enumerate(zip(self.i_conv_layers, self.k_conv_layers)):
            # residual convolution
            res_image = image
            res_image = res_image.permute(0, 3, 1, 2)
            res_image = i_conv_layer(res_image)
            res_image = res_image.permute(0, 2, 3, 1)
            image = image + res_image
            # data consistency layer
            cnn_fft = fft2(image)
            data_consistency_fourier = replace_values_on_mask_torch(cnn_fft, kspace, mask)
            data_consistency_fourier = data_consistency_fourier.permute(0, 3, 1, 2)
            data_consistency_fourier = k_conv_layer(data_consistency_fourier)
            data_consistency_fourier = data_consistency_fourier.permute(0, 2, 3, 1)
            image = ifft2(data_consistency_fourier)

        # equivalent of taking the module of the image
        image = complex_abs(image)
        image = center_crop(image, (320, 320))
        return image
