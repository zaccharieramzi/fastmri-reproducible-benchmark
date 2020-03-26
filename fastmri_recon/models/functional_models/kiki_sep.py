import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from ..training.compile import default_model_compile
from ..utils.complex import conv2d_complex
from ..utils.data_consistency import MultiplyScalar, enforce_kspace_data_consistency
from ..utils.fastmri_format import tf_fastmri_format
from ..utils.fourier import  tf_unmasked_adj_op, tf_unmasked_op


def kiki_sep_net(previous_net, multiply_scalar, input_size=(640, None, 1), n_convs=5, n_filters=16, noiseless=True, lr=1e-3, to_add='I', last=False, fastmri=True, activation='relu'):
    r"""This net is a sequence of convolution blocks in either image or k-space
        performed on top of a previous model whose weights will be frozen

    The original networks are described in [E2017]. It also features a data consistency
    layer before performing convolutions in the k-space.

    Parameters:
    previous_net (keras.models.Model or None): the previous model on top of which
        the convolutions defined here can be applied. If None, no previous model
        will be used, and the convolutions will be applied directly on the
        input.
    multiply_scalar (MultiplyScalar): the object common to all data consistency
        layers.
    input_size (tuple): the size of your input kspace, default to (640, None, 1)
    n_cascade (int): number of cascades, defaults to 2 like in paper
    n_convs (int): number of convolution in convolution blocks (N_I in paper), defaults to 5
    n_filters (int): number of filters in a convolution, defaults to 16
    noiseless (bool): whether the data consistency has to be done in a noiseless
        manner. If noiseless is `False`, the noise level is learned (i.e. lambda
        in paper, is learned). Defaults to `True`.
    lr (float): learning rate, defaults to 1e-3
    to_add (str): whether the convolutions happen in the image (I) or the k-space
        (K). This allows to apply the correct transformation and potentially
        data consistency.
    last (bool): whether to put the final image in fastMRI format, defaults
        to True (i.e. image will be cropped to 320, 320)
    activation (str or function): see https://keras.io/activations/ for info

    Returns:
    keras.models.Model: the partial KIKI net model, compiled
    """
    mask_shape = input_size[:-1]
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input')
    mask = Input(mask_shape, dtype='complex64', name='mask_input')
    if previous_net is None:
        kspace = conv2d_complex(kspace_input, n_filters, n_convs, output_shape=input_size, res=False, activation=activation, last_kernel_size=1)
        output = kspace
    else:
        previous_net.trainable = False
        if to_add == 'I':
            kspace = previous_net([kspace_input, mask])
            image = Lambda(tf_unmasked_adj_op, output_shape=input_size)(kspace)
            image = conv2d_complex(image, n_filters, n_convs, output_shape=input_size, res=True, activation=activation, last_kernel_size=1)
            output = image
            if last:
                if fastmri:
                    output = tf_fastmri_format(output)
                else:
                    output = Lambda(tf.math.abs)(output)
        elif to_add == 'K':
            image = previous_net([kspace_input, mask])
            kspace = Lambda(tf_unmasked_op, output_shape=input_size)(image)
            kspace = enforce_kspace_data_consistency(kspace, kspace_input, mask, input_size, multiply_scalar, noiseless)
            # K-net
            kspace = conv2d_complex(kspace, n_filters, n_convs, output_shape=input_size, res=False, activation=activation, last_kernel_size=1)
            output = kspace
    model = Model(inputs=[kspace_input, mask], outputs=output)
    default_model_compile(model, lr)
    return model


def full_kiki_net(input_size=(640, None, 1), **run_params):
    """The entire KIKI net model built with kiki_sep_net

    This function is useful to load the trained model weights.
    """
    multiply_scalar = MultiplyScalar()
    model = kiki_sep_net(None, multiply_scalar, input_size=input_size, to_add='K', last=False, **run_params)
    model = kiki_sep_net(model, multiply_scalar, input_size=input_size, to_add='I', last=False, **run_params)
    model = kiki_sep_net(model, multiply_scalar, input_size=input_size, to_add='K', last=False, **run_params)
    model = kiki_sep_net(model, multiply_scalar, input_size=input_size, to_add='I', last=True, **run_params)
    return model
