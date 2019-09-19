"""Deep cascade network."""
from keras.layers import Input, Lambda
from keras.models import Model
import tensorflow as tf
import torch
from torch import nn

from ..helpers.keras_utils import default_model_compile
from ..helpers.nn_mri import tf_fastmri_format, tf_unmasked_adj_op, tf_unmasked_op, replace_values_on_mask_torch, MultiplyScalar, conv2d_complex, enforce_kspace_data_consistency
from ..helpers.torch_utils import ConvBlock
from ..helpers.transforms import ifft2, fft2, center_crop, complex_abs


def cascade_net(input_size=(640, None, 1), n_cascade=5, n_convs=5, n_filters=16, noiseless=True, lr=1e-3, fastmri=True):
    r"""This net cascades several convolution blocks followed by data consistency layers

    The original network is described in [S2017]. Its implementation is
    available at https://github.com/js3611/Deep-MRI-Reconstruction in pytorch.

    Parameters:
    input_size (tuple): the size of your input kspace, default to (640, None, 1)
    n_cascade (int): number of cascades (n_c in paper), defaults to 5
    n_convs (int): number of convolution in convolution blocks (n_d + 1 in paper), defaults to 5
    n_filters (int): number of filters in a convolution, defaults to 16
    noiseless (bool): whether the data consistency has to be done in a noiseless
        manner. If noiseless is `False`, the noise level is learned (i.e. lambda
        in paper, is learned). Defaults to `True`.
    lr (float): learning rate, defaults to 1e-3
    fastmri (bool): whether to put the final image in fastMRI format, defaults
        to True (i.e. image will be cropped to 320, 320)

    Returns:
    keras.models.Model: the deep cascade net model, compiled
    """
    # inputs
    mask_shape = input_size[:-1]
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input')
    mask = Input(mask_shape, dtype='complex64', name='mask_input')

    zero_filled = Lambda(tf_unmasked_adj_op, output_shape=input_size, name='ifft_simple')(kspace_input)

    image = zero_filled
    multiply_scalar = MultiplyScalar()
    for i in range(n_cascade):
        # residual convolution
        image = conv2d_complex(image, n_filters, n_convs, output_shape=input_size, res=True)
        # data consistency layer
        kspace = Lambda(tf_unmasked_op, output_shape=input_size, name='fft_simple_{i}'.format(i=i+1))(image)
        kspace = enforce_kspace_data_consistency(kspace, kspace_input, mask, input_size, multiply_scalar, noiseless)
        image = Lambda(tf_unmasked_adj_op, output_shape=input_size, name='ifft_simple_{i}'.format(i=i+1))(kspace)
    # module and crop of image
    if fastmri:
        image = tf_fastmri_format(image)
    else:
        image = Lambda(tf.math.abs)(image)
    model = Model(inputs=[kspace_input, mask], outputs=image)

    default_model_compile(model, lr)

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
