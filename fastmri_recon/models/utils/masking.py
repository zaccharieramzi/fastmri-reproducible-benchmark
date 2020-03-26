import tensorflow as tf


def _mask_tf(x):
    k_data, mask = x
    mask = tf.expand_dims(tf.dtypes.cast(mask, k_data.dtype), axis=-1)
    masked_k_data = tf.math.multiply(mask, k_data)
    return masked_k_data
