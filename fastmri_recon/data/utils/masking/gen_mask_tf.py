import tensorflow as tf


def gen_mask_tf(kspace, accel_factor, multicoil=False, fixed_masks=False):
    shape = tf.shape(kspace)
    num_cols = shape[-1]
    center_fraction = (32 // accel_factor) / 100
    num_low_freqs = tf.dtypes.cast(num_cols, 'float32') * center_fraction
    num_low_freqs = tf.dtypes.cast((tf.round(num_low_freqs)), 'int32')
    prob = (num_cols / accel_factor - tf.dtypes.cast(num_low_freqs, 'float64')) / tf.dtypes.cast((num_cols - num_low_freqs), 'float64')
    if fixed_masks:
        tf.random.set_seed(0)
        seed = 0
    else:
        seed = None
    mask = tf.random.uniform(shape=tf.expand_dims(num_cols, axis=0), dtype='float64', seed=seed) < prob
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
    # we need the batch dimension for cases where we split the batch accross
    # multiple GPUs
    if multicoil:
        final_mask_reshaped = tf.tile(final_mask_reshaped, [shape[0], 1, 1, 1])
    else:
        final_mask_reshaped = tf.tile(final_mask_reshaped, [shape[0], 1, 1])
    fourier_mask = tf.cast(final_mask_reshaped, tf.uint8)
    return fourier_mask

def gen_mask_equidistant_tf(kspace, accel_factor, multicoil=False):
    shape = tf.shape(kspace)
    num_cols = shape[-1]
    center_fraction = (32 // accel_factor) / 100
    num_low_freqs = tf.cast(num_cols, tf.float32) * center_fraction
    num_low_freqs = tf.cast(tf.round(num_low_freqs), tf.int32)
    num_high_freqs = num_cols // accel_factor - num_low_freqs
    high_freqs_spacing = (num_cols - num_low_freqs) // num_high_freqs
    acs_lim = (num_cols - num_low_freqs + 1) // 2
    mask_offset = tf.random.uniform([], maxval=high_freqs_spacing, dtype=tf.int32)
    high_freqs_location = tf.range(mask_offset, num_cols, high_freqs_spacing)
    low_freqs_location = tf.range(acs_lim, acs_lim + num_low_freqs)
    mask_locations = tf.concat([high_freqs_location, low_freqs_location], 0)
    mask = tf.scatter_nd(
        mask_locations[:, None],
        tf.ones(mask_locations.shape)[:, None],
        [num_cols, 1],
    )
    final_mask = tf.where(
        tf.cast(mask, tf.bool),
        tf.ones_like(mask),
        mask,
    )
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
    # we need the batch dimension for cases where we split the batch accross
    # multiple GPUs
    if multicoil:
        final_mask_reshaped = tf.tile(final_mask_reshaped, [shape[0], 1, 1, 1])
    else:
        final_mask_reshaped = tf.tile(final_mask_reshaped, [shape[0], 1, 1])
    fourier_mask = tf.cast(final_mask_reshaped, tf.uint8)
    return fourier_mask
