import tensorflow as tf


def keras_psnr(y_true, y_pred):
    max_pixel = tf.math.reduce_max(y_true)
    min_pixel = tf.math.reduce_min(y_true)
    return tf.image.psnr(y_true, y_pred, max_pixel - min_pixel)

def keras_ssim(y_true, y_pred):
    max_pixel = tf.math.reduce_max(y_true)
    min_pixel = tf.math.reduce_min(y_true)
    return tf.image.ssim(y_true, y_pred, max_pixel - min_pixel)
