import tensorflow as tf

from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet


def test_ncpdnet_init_and_call(ktraj):
    model = NCPDNet(n_iter=3, n_primal=3, n_filters=8)
    image_shape = (640, 372)
    nspokes = 15
    traj = ktraj(image_shape, nspokes)
    spokelength = image_shape[-1] * 2
    kspace_shape = spokelength * nspokes
    model([
        tf.zeros([1, 1, kspace_shape], dtype=tf.complex64),
        traj,
        tf.constant([image_shape[-1]])
    ])
