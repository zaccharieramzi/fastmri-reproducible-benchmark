import tensorflow as tf


def gen_mask_tf(kspace, accel_factor, multicoil=False):
    shape = tf.shape(kspace)
    num_cols = shape[-1]
    center_fraction = (32 // accel_factor) / 100
    num_low_freqs = tf.dtypes.cast(num_cols, 'float32') * center_fraction
    num_low_freqs = tf.dtypes.cast((tf.round(num_low_freqs)), 'int32')
    prob = (num_cols / accel_factor - tf.dtypes.cast(num_low_freqs, 'float64')) / tf.dtypes.cast((num_cols - num_low_freqs), 'float64')
    mask = tf.random.uniform(shape=tf.expand_dims(num_cols, axis=0), dtype='float64') < prob
    pad = (num_cols - num_low_freqs + 1) // 2
    final_mask = tf.concat([
        mask[:pad],
        tf.ones([num_low_freqs], dtype=tf.bool),
        mask[pad+num_low_freqs:],
    ], axis=0)

    # Reshape the mask
    mask_shape = tf.ones_like(shape)
    if multicoil:
        mask_shape = mask_shape[:3]
    else:
        mask_shape = mask_shape[:2]
    final_mask_shape = tf.concat([
        mask_shape,
        tf.expand_dims(num_cols, axis=0),
    ], axis=0)
    final_mask_reshaped = tf.reshape(final_mask, final_mask_shape)
    fourier_mask = tf.cast(final_mask_reshaped, tf.uint8)
    return fourier_mask
