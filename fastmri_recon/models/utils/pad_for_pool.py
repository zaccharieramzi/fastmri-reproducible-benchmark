import tensorflow as tf


def pad_for_pool(inputs, n_pools):
    problematic_dim = tf.shape(inputs)[-2]
    k = tf.math.floordiv(problematic_dim, 2 ** n_pools)
    n_pad = tf.cond(
        tf.math.floormod(problematic_dim, 2 ** n_pools) == 0,
        lambda: 0,
        lambda: (k + 1) * 2 ** n_pools - problematic_dim,
    )
    paddings = [
        (0, 0),
        (0, 0),  # here in the context of fastMRI there is nothing to worry about because the dim is 640 (128 x 5)
        (n_pad//2, n_pad//2),
        (0, 0),
    ]
    inputs_padded = tf.pad(inputs, paddings)
    return inputs_padded, n_pad
