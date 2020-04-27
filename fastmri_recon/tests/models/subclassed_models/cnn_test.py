import pytest
import tensorflow as tf

from fastmri_recon.models.subclassed_models.cnn import CNNComplex


@pytest.mark.parametrize('model_kwargs', [
    {},
    {'res': False},
])
def test_cnn_complex_init_call(model_kwargs):
    model = CNNComplex(**model_kwargs)
    model(tf.zeros([1, 640, 320, 1], dtype=tf.complex64))
