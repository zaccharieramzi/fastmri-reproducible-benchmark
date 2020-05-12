import pytest
import tensorflow as tf

from fastmri_recon.models.subclassed_models.updnet import UPDNet


@pytest.mark.parametrize('model_kwargs, n_phase_encoding', [
    ({}, 320),
    ({}, 550),
    ({'primal_only': True}, 320),
])
def test_updnet_init_and_call(model_kwargs, n_phase_encoding):
    model = UPDNet(**model_kwargs)
    model([
        tf.zeros([1, 640, n_phase_encoding, 1], dtype=tf.complex64),  # kspace
        tf.zeros([1, 640, n_phase_encoding], dtype=tf.complex64),  # mask
    ])

@pytest.mark.parametrize('model_kwargs', [
    {},
    {'channel_attention_kwargs': {'dense': True}},
    {'refine_smaps': True},
    {
        'channel_attention_kwargs': {'dense': True},
        'refine_smaps': True,
        'non_linearity': 'prelu',
    }
])
def test_updnet_multicoil_init_and_call(model_kwargs):
    model = UPDNet(primal_only=True, multicoil=True, **model_kwargs)
    model([
        tf.zeros([1, 5, 640, 320, 1], dtype=tf.complex64),  # kspace
        tf.zeros([1, 5, 640, 320], dtype=tf.complex64),  # mask
        tf.zeros([1, 5, 640, 320], dtype=tf.complex64),  # smaps
    ])
