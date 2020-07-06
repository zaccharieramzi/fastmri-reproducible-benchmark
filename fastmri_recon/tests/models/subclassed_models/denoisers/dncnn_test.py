import tensorflow as tf

from fastmri_recon.models.subclassed_models.denoisers.dncnn import DnCNN


def test_dncnn():
    n_out = 2
    model = DnCNN(n_convs=2, n_filters=4, res=False, n_outputs=n_out)
    res = model(tf.zeros([1, 32, 32, 4]))
    assert res.shape[-1] == n_out
