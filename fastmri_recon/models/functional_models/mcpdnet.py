"""Learned Primal-dual network adapted to MRI for multi-contrast."""
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, concatenate, Add
from tensorflow.keras.models import Model

from ..training.compile import default_model_compile
from ..utils.complex import conv2d_complex
from ..utils.fastmri_format import tf_fastmri_format
from ..utils.fourier import mc_tf_adj_op, mc_tf_op


def mcpdnet(input_size=(None, None, None, 1), n_filters=32, lr=1e-3, n_primal=5, n_dual=5, n_iter=10, primal_only=False, fastmri=True, activation='relu'):
    r"""This net unrolls the PDHG algorithm in the context of MRI for multi-contrast
    It is a extended version of https://github.com/zaccharieramzi/fastmri-reproducible-benchmark/blob/master/fastmri_recon/models/functional_models/pdnet.py 
    in context of multi-contrast.
    

    Parameters:
    input_size (tuple): the size of your input kspace, defaults to (None, None, None, 1) - (mc X H X W X 1)
    n_filters (int): number of filters in the convolution blocks, defaults to 32
    lr (float): learning rate, defaults to 1e-3
    n_primal (int): number of elements in the primal memory buffer, defaults to 5
    n_dual (int): number of elements in the dual memory buffer, defaults to 5
    n_iter (int): number of PDHG unrolled iterations, defaults to 10
    primal_only (bool): whether to use non linearity in the dual space, or just
        use the residual, defaults to False
    fastmri (bool): whether to put the final image in fastMRI format, defaults
        to True (i.e. image will be cropped to 320, 320)
    activation (str or function): see https://keras.io/activations/ for info

    Returns:
    keras.models.Model: the multi-contrast primal dual net model, compiled
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
        dual_eval_exp = Lambda(mc_tf_op, output_shape=input_size, arguments={'idx': 1}, name='fft_masked_{i}'.format(i=i+1))([primal, mask]) # multicontrast-fourier transform
        if primal_only:
            dual = Lambda(lambda x: x[0] - x[1], output_shape=input_size, name='dual_residual_{i}'.format(i=i+1))([dual_eval_exp, kspace_input])
        else:
            update = concatenate([dual, dual_eval_exp, kspace_input], axis=-1)
            update = conv2d_complex(update, n_filters, 2, output_shape=dual_shape, res=False, activation=activation)
            dual = Add()([dual, update])


        # Then work in image space (primal space)
        primal_exp = Lambda(mc_tf_adj_op, output_shape=input_size, name='ifft_masked_{i}'.format(i=i+1))([dual, mask]) #multicontrast inverse fourier transform
        update = concatenate([primal, primal_exp], axis=-1)
        update = conv2d_complex(update, n_filters, 2, output_shape=primal_shape, res=False, activation=activation)
        primal = Add()([primal, update])

    # formatting the image and finalizing the model definition
    image_res = Lambda(lambda x: x[..., 0:1], output_shape=input_size, name='image_getting')(primal)
    if fastmri:
        image_res = tf_fastmri_format(image_res)
    else:
        image_res = Lambda(tf.math.abs)(image_res)
    model = Model(inputs=[kspace_input, mask], outputs=image_res)
    default_model_compile(model, lr)


    return model

