import tensorflow as tf
from tensorflow.python.ops.signal.fft_ops import fft2d, ifft2d, ifftshift, fftshift


def _compute_scaling_norm(x):
    image_area = tf.reduce_prod(tf.shape(x)[-2:])
    image_area = tf.cast(image_area, 'float32')
    scaling_norm = tf.sqrt(image_area)
    scaling_norm = tf.cast(scaling_norm, x.dtype)
    return scaling_norm

def _order_for_ft(x):
    return tf.transpose(x, [0, 3, 1, 2])

def _order_after_ft(x):
    return tf.transpose(x, [0, 2, 3, 1])

def ortho_fft2d(image):
    image = _order_for_ft(image)
    shift_axes = [2, 3]
    scaling_norm = _compute_scaling_norm(image)
    shifted_image = fftshift(image, axes=shift_axes)
    kspace_shifted = fft2d(shifted_image)
    kspace_unnormed = ifftshift(kspace_shifted, axes=shift_axes)
    kspace = kspace_unnormed / scaling_norm
    kspace = _order_after_ft(kspace)
    return kspace

def ortho_ifft2d(kspace):
    kspace = _order_for_ft(kspace)
    shift_axes = [2, 3]
    scaling_norm = _compute_scaling_norm(kspace)
    shifted_kspace = ifftshift(kspace, axes=shift_axes)
    image_shifted = ifft2d(shifted_kspace)
    image_unnormed = fftshift(image_shifted, axes=shift_axes)
    image = image_unnormed * scaling_norm
    image = _order_after_ft(image)
    return image
