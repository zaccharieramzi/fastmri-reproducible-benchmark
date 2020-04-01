import pytest
import tensorflow as tf

from fastmri_recon.models.subclassed_models.single_coil.pdnet import PDNet


@pytest.mark.parametrize('model_kwargs', [
    {},
    {'primal_only': False}
])
def test_pdnet_init_and_call(model_kwargs):
    model = PDNet(**model_kwargs)
    model([
        tf.zeros([1, 640, 320, 1], dtype=tf.complex64),  # kspace
        tf.zeros([1, 640, 320], dtype=tf.complex64),  # mask
    ])
