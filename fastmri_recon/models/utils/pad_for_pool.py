import tensorflow as tf


def pad_for_pool(inputs, n_pools):
    inputs_padded, paddings = pad_for_pool_whole_plane(inputs, n_pools)
    return inputs_padded, paddings[-1]

def pad_for_pool_whole_plane(inputs, n_pools):
    problematic_dims = tf.shape(inputs)[-3:-1]
    k = tf.math.floordiv(problematic_dims, 2 ** n_pools)
    n_pad = tf.where(
        tf.math.equal(tf.math.mod(problematic_dims, 2 ** n_pools), 0),
        0,
        (k+1)* 2**n_pools - problematic_dims
    )
    padding_left = tf.where(
        tf.logical_or(
            tf.math.equal(tf.math.mod(problematic_dims, 2), 0),
            tf.math.equal(n_pad, 0),
        ),
        n_pad//2,
        n_pad//2 + 1,
    )
    padding_right = n_pad//2
    paddings_short = [(padding_left[i], padding_right[i]) for i in range(2)]
    paddings = [
        (0, 0),
        paddings_short[0],
        paddings_short[1],
        (0, 0),
    ]
    inputs_padded = tf.pad(inputs, paddings)
    return inputs_padded, paddings_short
