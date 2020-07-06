import tensorflow as tf

from fastmri_recon.models.subclassed_models.denoisers.focnet import FocNet


def test_focnet():
    n_out = 2
    model = FocNet(
        n_filters=4,
        n_outputs=n_out,
    )
    res = model(tf.zeros([1, 32, 32, 4]))
    assert res.shape[-1] == n_out
