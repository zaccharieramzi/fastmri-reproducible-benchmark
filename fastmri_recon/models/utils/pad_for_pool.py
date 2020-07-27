import tensorflow as tf


def pad_for_pool(inputs, n_pools):
    problematic_dim = tf.shape(inputs)[-2]
    k = tf.math.floordiv(problematic_dim, 2 ** n_pools)
    n_pad = tf.cond(
        tf.math.mod(problematic_dim, 2 ** n_pools) == 0,
        lambda: 0,
        lambda: (k + 1) * 2 ** n_pools - problematic_dim,
    )
    padding = tf.cond(
        tf.math.mod(problematic_dim, 2) == 0,
        lambda: (n_pad//2, n_pad//2),
        lambda: (n_pad//2 + 1, n_pad//2),
    )
    paddings = [
        (0, 0),
        (0, 0),  # here in the context of fastMRI there is nothing to worry about because the dim is 640 (128 x 5)
        # even for brain data, it shouldn't be a problem, since it's 640, 512, or 768.
        padding,
        (0, 0),
    ]
    inputs_padded = tf.pad(inputs, paddings)
    return inputs_padded, padding
