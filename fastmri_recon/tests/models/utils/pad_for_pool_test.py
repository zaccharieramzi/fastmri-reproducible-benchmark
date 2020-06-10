import pytest
import tensorflow as tf

from fastmri_recon.models.utils.pad_for_pool import pad_for_pool


@pytest.mark.parametrize('dim, n_pools, n_pad_expected',[
    (454, 2, 2),
    (454, 1, 0),
    (320, 3, 0),
    (322, 3, 6),
    (372, 4, 12),
])
def test_pad_for_pool(dim, n_pools, n_pad_expected):
    inputs = tf.zeros([1, 640, dim, 1])
    _, n_pad = pad_for_pool(inputs, n_pools)
    tf_tester = tf.test.TestCase()
    tf_tester.assertEqual(n_pad, n_pad_expected)
