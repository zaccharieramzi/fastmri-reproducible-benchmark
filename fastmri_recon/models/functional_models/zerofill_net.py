"""Simple example-debug net."""
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from ..training.compile import default_model_compile
from ..utils.fastmri_format import tf_fastmri_format
from ..utils.fourier import tf_adj_op


def zerofill_net(input_size=(640, None, 1), **dummy_kwargs):
    """A net that performs a simple zero-filled reconstruction

    Parameters:
    input_size (tuple): the size of your input kspace

    Returns:
    keras.models.Model: the zerofill net model, compiled
    """
    # shapes
    mask_shape = input_size[:-1]
    # inputs and buffers
    kspace_input = Input(input_size, dtype='complex64', name='kspace_input_simple')
    mask = Input(mask_shape, dtype='complex64', name='mask_input_simple')
    # # simple inverse
    image_res = Lambda(tf_adj_op, output_shape=input_size, name='ifft_simple')([kspace_input, mask])
    image_res = tf_fastmri_format(image_res)
    model = Model(inputs=[kspace_input, mask], outputs=image_res)
    default_model_compile(model, lr=1e-3)


    return model
