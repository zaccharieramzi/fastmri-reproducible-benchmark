import pytest
import tensorflow as tf

from fastmri_recon.models.subclassed_models.unet import UnetComplex


@pytest.mark.parametrize('model_kwargs', [
    {},
    {'n_input_channels': 6},
    {'res': True},
    {'non_linearity': 'prelu'},
    {'channel_attention_kwargs': {'dense': True}},
])
def test_cnn_complex_init_call(model_kwargs):
    model = UnetComplex(**model_kwargs)
    model(tf.zeros(
        [1, 640, 320, model_kwargs.get('n_input_channels', 1)],
        dtype=tf.complex64,
    ))
