"""Learned Primal-dual network adapted to MRI."""
from keras.layers import Input, Lambda, concatenate, Add
from keras.models import Model
import tensorflow as tf
import torch
from torch import nn

from ..helpers.keras_utils import default_model_compile
from ..helpers.nn_mri import tf_fastmri_format, tf_adj_op, tf_op, conv2d_complex
from ..helpers.torch_utils import ConvBlock
from ..helpers.transforms import ifft2, fft2, center_crop, complex_abs


def pdnet(input_size=(640, None, 1), n_filters=32, lr=1e-3, n_primal=5, n_dual=5, n_iter=10, primal_only=False):
    r"""This net unrolls the PDHG algorithm in the context of MRI

    The original network is described in [A2017]. Its implementation is
    available at https://github.com/adler-j/learned_primal_dual. A network
    adapted to MRI has been re-implemented in [C2019] without known
    public implementation. Here the Radon transform operator is replaced by the
    Fourier Transform operator. Because of the needs of this particular
    dataset (the size of the kspaces is not constant) we can't stick to an odl
    implementation for the operator. We therefore stick to the TensorFlow
    defined FFT. It also has the significant advantage to be able to leverage
    CuFFT.
    Because the kspace and image are complex numbers we simply need to adapt
    the convolutions. The real and imaginary part of either are concatenated
    along the last dimension, making a 2-channel image that we can then flow
    through the convolutions. We then 're-complexify' the last output of the
    convolutions.

    Parameters:
    input_size (tuple): the size of your input kspace, defaults to (640, None, 1)
    n_filters (int): number of filters in the convolution blocks, defaults to 32
    lr (float): learning rate, defaults to 1e-3
    n_primal (int): number of elements in the primal memory buffer, defaults to 5
    n_dual (int): number of elements in the dual memory buffer, defaults to 5
    n_iter (int): number of PDHG unrolled iterations, defaults to 10
    primal_only (bool): whether to use non linearity in the dual space, or just
        use the residual, defaults to False

    Returns:
    keras.models.Model: the primal dual net model, compiled
    """
    # shapes
    mask_shape = input_size[:-1]
    primal_shape = list(input_size)
    primal_shape[-1] = n_primal
    primal_shape = tuple(primal_shape)
    dual_shape = list(input_size)
    dual_shape[-1] = n_dual
    dual_shape = tuple(dual_shape)

    # inputs and buffers
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input')
    mask = Input(mask_shape, dtype='complex64', name='mask_input')
    primal = Lambda(lambda x: tf.concat([tf.zeros_like(x, dtype='complex64')] * n_primal, axis=-1), output_shape=primal_shape, name='buffer_primal')(kspace_input)
    if not primal_only:
        dual = Lambda(lambda x: tf.concat([tf.zeros_like(x, dtype='complex64')] * n_dual, axis=-1), output_shape=dual_shape, name='buffer_dual')(kspace_input)

    # unrolled iterations
    for i in range(n_iter):
        # first work in kspace (dual space)
        dual_eval_exp = Lambda(tf_op, output_shape=input_size, arguments={'idx': 1}, name='fft_masked_{i}'.format(i=i+1))([primal, mask])
        if primal_only:
            dual = Lambda(lambda x: x[0] - x[1], output_shape=input_size, name='dual_residual_{i}'.format(i=i+1))([dual_eval_exp, kspace_input])
        else:
            update = concatenate([dual, dual_eval_exp, kspace_input], axis=-1)
            update = conv2d_complex(update, n_filters, 2, output_shape=dual_shape, res=False)
            dual = Add()([dual, update])


        # Then work in image space (primal space)
        primal_exp = Lambda(tf_adj_op, output_shape=input_size, name='ifft_masked_{i}'.format(i=i+1))([dual, mask])
        update = concatenate([primal, primal_exp], axis=-1)
        update = conv2d_complex(update, n_filters, 2, output_shape=primal_shape, res=False)
        primal = Add()([primal, update])

    # formatting the image and finalizing the model definition
    image_res = Lambda(lambda x: x[..., 0:1], output_shape=input_size, name='image_getting')(primal)
    image_res = tf_fastmri_format(image_res)
    model = Model(inputs=[kspace_input, mask], outputs=image_res)
    default_model_compile(model, lr)


    return model


class PDNet(torch.nn.Module):
    def __init__(self, n_filters=32, n_primal=5, n_dual=5, n_iter=10, primal_only=False):
        super(PDNet, self).__init__()
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
