import tensorflow as tf
from tensorflow.python.ops.signal.fft_ops import ifft2d, ifftshift, fftshift


def extract_smaps(kspace, low_freq_percentage=8, background_thresh=4e-6):
    n_low_freq = tf.cast(tf.shape(kspace)[-2:] * low_freq_percentage / 100, tf.int32)
    center_dimension = tf.cast(tf.shape(kspace)[-2:] / 2, tf.int32)
    low_freq_lower_locations = center_dimension - tf.cast(n_low_freq / 2, tf.int32)
    low_freq_upper_locations = center_dimension + tf.cast(n_low_freq / 2, tf.int32)
    ###
    # NOTE: the following stands for in numpy:
    # low_freq_mask = np.zeros_like(kspace)
    # low_freq_mask[
    #     ...,
    #     low_freq_lower_locations[0]:low_freq_upper_locations[0],
    #     low_freq_lower_locations[1]:low_freq_upper_locations[1]
    # ] = 1
    x_range = tf.range(low_freq_lower_locations[0], low_freq_upper_locations[0])
    y_range = tf.range(low_freq_lower_locations[1], low_freq_upper_locations[1])
    X_range, Y_range = tf.meshgrid(x_range, y_range)
    X_range = tf.reshape(X_range, (-1,))
    Y_range = tf.reshape(Y_range, (-1,))
    low_freq_mask_indices = tf.stack([X_range, Y_range], axis=-1)
    # we have to transpose because only the first dimension can be indexed in
    # scatter_nd
    scatter_nd_perm = [2, 3, 0, 1]
    low_freq_mask = tf.scatter_nd(
        indices=low_freq_mask_indices,
        updates=tf.ones([
            tf.size(X_range),
            tf.shape(kspace)[0],
            tf.shape(kspace)[1]],
        ),
        shape=[tf.shape(kspace)[i] for i in scatter_nd_perm],
    )
    low_freq_mask = tf.transpose(low_freq_mask, perm=scatter_nd_perm)
    ###
    low_freq_kspace = kspace * tf.cast(low_freq_mask, kspace.dtype)
    shifted_kspace = ifftshift(low_freq_kspace, axes=[2, 3])
    coil_image_low_freq_shifted = ifft2d(shifted_kspace)
    coil_image_low_freq = fftshift(coil_image_low_freq_shifted, axes=[2, 3])
    # no need to norm this since they all have the same norm
    low_freq_rss = tf.norm(coil_image_low_freq, axis=1)
    coil_smap = coil_image_low_freq / low_freq_rss[:, None]
    # for now we do not perform background removal based on low_freq_rss
    # could be done with 1D k-means or fixed background_thresh, with tf.where
    return coil_smap
