import tensorflow as tf


def tf_af(mask):
    mask_int = tf.dtypes.cast(mask, 'int32')
    return tf.shape(mask_int)[0] / tf.reduce_sum(mask_int)
