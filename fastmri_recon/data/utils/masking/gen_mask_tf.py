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
    final_mask_shape = tf.concat([
        mask_shape[:2],
        tf.expand_dims(num_cols, axis=0),
    ], axis=0)
    final_mask_reshaped = tf.reshape(final_mask, final_mask_shape)
    tiling = [shape[0], shape[1], 1]
    if multicoil:
        tiling = [shape[0], shape[1], shape[2], 1]
    fourier_mask = tf.tile(final_mask_reshaped, tiling)
    fourier_mask = tf.dtypes.cast(fourier_mask, 'complex64')
    return fourier_mask
