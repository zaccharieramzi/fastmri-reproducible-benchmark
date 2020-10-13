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

def gen_mask_equidistant_tf(kspace, accel_factor, multicoil=False, mask_type='real'):
    shape = tf.shape(kspace)
    num_cols = shape[-1]
    center_fraction = (32 // accel_factor) / 100
    num_low_freqs = tf.cast(num_cols, tf.float32) * center_fraction
    num_low_freqs = tf.cast(tf.round(num_low_freqs), tf.int32)
    acs_lim = (num_cols - num_low_freqs + 1) // 2
    if mask_type == 'real':
        num_high_freqs = num_cols // accel_factor - num_low_freqs
        high_freqs_spacing = (num_cols - num_low_freqs) // num_high_freqs
        mask_offset = tf.random.uniform([], maxval=high_freqs_spacing, dtype=tf.int32)
        high_freqs_location = tf.range(mask_offset, num_cols, high_freqs_spacing)
    else:
        adjusted_accel = (accel_factor * (num_low_freqs - num_cols)) / (num_low_freqs * accel_factor - num_cols)
        adjusted_accel_round = tf.cast(tf.round(adjusted_accel), tf.int32)
        mask_offset = tf.random.uniform([], maxval=tf.round(adjusted_accel_round), dtype=tf.int32)
        high_freqs_location = tf.range(mask_offset, num_cols, adjusted_accel)
        high_freqs_location = tf.cast(tf.round(high_freqs_location), tf.int32)
        high_freqs_location = tf.minimum(high_freqs_location, num_cols-1)
    low_freqs_location = tf.range(acs_lim, acs_lim + num_low_freqs)
    mask_locations = tf.concat([high_freqs_location, low_freqs_location], 0)
    mask = tf.scatter_nd(
        mask_locations[:, None],
        tf.ones(tf.shape(mask_locations))[:, None],
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
