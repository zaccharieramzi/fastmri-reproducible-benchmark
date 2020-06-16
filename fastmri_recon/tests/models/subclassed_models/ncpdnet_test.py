import tensorflow as tf

from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet


def test_ncpdnet_init_and_call():
    model = NCPDNet()
    model([
        tf.zeros([1, 640, 320, 1], dtype=tf.complex64),  # kspace
        tf.zeros([1, 640, 320], dtype=tf.complex64),  # mask
    ])
