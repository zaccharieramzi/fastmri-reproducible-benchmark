import tensorflow as tf

from fastmri_recon.models.subclassed_models.denoisers.didn import DIDN


def test_didn():
    n_out = 2
    model = DIDN(
        n_filters=4,
        n_dubs=2,
        n_convs_recon=2,
        n_outputs=n_out,
    )
    res = model(tf.zeros([1, 32, 32, 4]))
    assert res.shape[-1] == n_out
