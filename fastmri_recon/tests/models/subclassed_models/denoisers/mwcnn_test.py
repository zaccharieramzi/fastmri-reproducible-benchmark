import tensorflow as tf

from fastmri_recon.models.subclassed_models.denoisers.mwcnn import MWCNN


def test_mwcnn():
    n_out = 2
    model = MWCNN(
        n_scales=3,
        kernel_size=3,
        bn=False,
        n_filters_per_scale=[4, 8, 8],
        n_convs_per_scale=[2, 2, 2],
        n_first_convs=2,
        first_conv_n_filters=4,
        res=False,
        n_outputs=n_out,
    )
    res = model(tf.zeros([1, 32, 32, 4]))
    assert res.shape[-1] == n_out
