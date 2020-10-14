import tensorflow as tf

from fastmri_recon.models.subclassed_models.vnet import Vnet


def test_dncnn():
    n_out = 2
    model = Vnet(layers_n_non_lins=2, layers_n_channels=[4, 8], n_output_channels=n_out)
    res = model(tf.zeros([1, 32, 32, 32, 4]))
    assert res.shape[-1] == n_out
