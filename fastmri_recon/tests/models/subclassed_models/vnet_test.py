import tensorflow as tf

from fastmri_recon.models.subclassed_models.vnet import Vnet


def test_vnet():
    n_out = 2
    model = Vnet(layers_n_non_lins=2, layers_n_channels=[4, 8], n_output_channels=n_out)
    input_shape = [1, 176, 64, 64, 4]
    res = model(tf.zeros(input_shape))
    assert res.shape[-1] == n_out
    assert res.shape[:2] == input_shape[:2]
