import numpy as np
import tensorflow as tf


def crop_center(img, cropx, cropy=None):
    # taken from https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image/39382475
    if cropy is None:
        cropy = cropx
    y, x = img.shape
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def gen_mask(kspace, accel_factor=8, seed=None):
    # inspired by https://github.com/facebookresearch/fastMRI/blob/master/common/subsample.py
    shape = kspace.shape
    num_cols = shape[-1]

    center_fraction = (32 // accel_factor) / 100
    acceleration = accel_factor

    # Create the mask
    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    mask = np.random.default_rng(seed).uniform(size=num_cols) < prob
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = True

    # Reshape the mask
    mask_shape = [1 for _ in shape]
    mask_shape[-1] = num_cols
    mask = mask.reshape(*mask_shape)
    return mask

def gen_mask_tf(kspace, accel_factor):
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
    fourier_mask = tf.tile(final_mask_reshaped, [shape[0], shape[1], 1])
    fourier_mask = tf.dtypes.cast(fourier_mask, 'complex64')
    return fourier_mask

def tf_af(mask):
    mask_int = tf.dtypes.cast(mask, 'int32')
    return tf.shape(mask_int)[0] / tf.reduce_sum(mask_int)

def gen_mask_vd(kspace, accel_factor=8):
    shape = kspace.shape
    num_cols = shape[-1]

    center_fraction = (32 // accel_factor) / 100
    acceleration = accel_factor

    # Create the mask
    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - 3 * num_low_freqs)
    num_sampling = int(prob * num_cols)
    selected_indexes = np.random.default_rng().randint(0, num_cols, size=(num_sampling, 2)).mean(axis=-1).astype('int')
    mask = np.zeros((num_cols,)).astype('bool')
    mask[selected_indexes] = True
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = True

    # Reshape the mask
    mask_shape = [1 for _ in shape]
    mask_shape[-1] = num_cols
    mask = mask.reshape(*mask_shape)
    return mask

# taken from https://github.com/facebookresearch/fastMRI/
def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (array-like): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        array-like: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (array-like): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            array-like: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std

# TODO have the same in 3D
def keras_psnr(y_true, y_pred):
    max_pixel = tf.math.reduce_max(y_true)
    min_pixel = tf.math.reduce_min(y_true)
    return tf.image.psnr(y_true, y_pred, max_pixel - min_pixel)

def keras_ssim(y_true, y_pred):
    max_pixel = tf.math.reduce_max(y_true)
    min_pixel = tf.math.reduce_min(y_true)
    return tf.image.ssim(y_true, y_pred, max_pixel - min_pixel)
