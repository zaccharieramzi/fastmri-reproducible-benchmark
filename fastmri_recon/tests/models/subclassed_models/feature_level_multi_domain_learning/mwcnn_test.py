import numpy as np
import tensorflow as tf

from from fastmri_recon.models.subclassed_models.feature_level_multi_domain_learning.mwcnn import MWCNNMultiDomain


def test_unet():
    n_out = 2
    model = MWCNNMultiDomain(
        n_filters_per_scale=[4, 8, 16],
        n_convs_per_scale=[2, 2, 2],
        n_first_convs=2,
        first_conv_n_filters=4,
        n_outputs=n_out,
    )
    shape = [1, 32, 32, 4]
    res = model(tf.zeros(shape))
    out_shape = [*shape[:-1], n_out]
    assert res.shape.as_list() == out_shape


def test_unet_change():
    model = MWCNNMultiDomain(
        n_filters_per_scale=[4, 8, 16],
        n_convs_per_scale=[2, 2, 2],
        n_first_convs=2,
        first_conv_n_filters=4,
        n_outputs=2,
    )
    x = tf.random.normal((1, 64, 64, 2))
    y = x
    model(x)
    before = [v.numpy() for v in model.trainable_variables]
    model.compile(optimizer='sgd', loss='mse')
    model.train_on_batch(x, y)
    after = [v.numpy() for v in model.trainable_variables]
    for b, a in zip(before, after):
        assert np.any(np.not_equal(b, a))
